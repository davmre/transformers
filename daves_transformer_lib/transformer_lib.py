from typing import Callable, List

import jax
from jax import numpy as jnp

from flax import linen as nn


class EncoderDecoder(nn.Module):

    encoder: nn.Module
    decoder: nn.Module
    src_embed: Callable
    tgt_embed: Callable
    generator: nn.Module

    def __call__(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):

    #d_model: int
    #vocab: Any  # TODO: what is this and how to build it??

    proj: nn.Module

    def __call__(self, x):
        return jax.nn.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    features: int
    num_layers: int
    layer_fn: Callable

    def setup(self):
        self.layers = [self.layer_fn() for _ in range(self.num_layers)]
        self.norm = LayerNorm(self.features)

    def __call__(self, x, mask):
        for layer in self.layers:
            x = self.layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    features: int
    eps: float = 1e-6

    def setup(self):
        self.a_2 = self.param('a_2', lambda k, s: jnp.ones(s), (self.features,))
        self.b_2 = self.param('b_2', lambda k, s: jnp.zeros(s),
                              (self.features,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1)
        std = jnp.std(x, axis=-1)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    size: int
    dropout_rate: float

    def setup(self):
        self.norm = LayerNorm(self.size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, sublayer: Callable):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    size: int
    self_attn: nn.Module
    feed_forward: nn.Module
    dropout_rate: float

    def setup(self):
        self.attn_sublayer = SublayerConnection(self.size,
                                                dropout_rate=self.dropout_rate)
        self.feed_forward_sublayer = SublayerConnection(
            self.size, dropout_rate=self.dropout_rate)

    def __call__(self, x, mask):
        x = self.attn_sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return self.feed_forward_sublayer(x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(float(d_k))
    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)
    p_attn = jax.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return jnp.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    h: int
    d_model: int
    dropout_rate: float = 0.1

    def setup(self):
        self.d_k = self.d_model // self.h
        [
            self.query_linear, self.key_linear, self.value_linear,
            self.final_linear
        ] = [
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform)
            for _ in range(4)
        ]
        self.attn = None
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all `h` heads.
            mask = mask[:, None, :]

        # q, k, v all have shape [n, h * d_k].
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        # Convert to shape [h, n, d_k].
        query, key, value = [
            jnp.transpose(jnp.reshape(t, [-1, self.h, self.d_k]), (0, 1))
            for t in (query, key, value)
        ]

        x, self.attn = attention(query, key, value, dropout=self.dropout)
        # x shape [h, n, d_k] -> [n, h * d_k]
        x = jnp.reshape(jnp.transpose(x, (0, 1)), (-1, self.h * self.d_k))
        return self.final_linear(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    def setup(self):
        self.w_1 = nn.Dense(self.d_ff,
                            kernel_init=nn.initializers.xavier_uniform)
        self.w_2 = nn.Dense(self.d_model,
                            kernel_init=nn.initializers.xavier_uniform)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _call__(self, x):
        return self.w_2(self.dropout(jax.nn.relu(self.w_1(x))))


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout_rate=0.1):
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        Encoder(features=d_model,
                num_layers=N,
                layer_fn=lambda: EncoderLayer(
                    size=d_model,
                    self_attn=MultiHeadedAttention(h, d_model),
                    feed_forward=PositionwiseFeedForward(
                        d_model, d_ff, dropout_rate),
                    dropout_rate=dropout_rate)),
        Decoder(features=d_model,
                num_layers=N,
                layer_fn=lambda: DecoderLayer(
                    size=d_model,
                    self_attn=MultiHeadedAttention(h, d_model),
                    feed_forward=PositionwiseFeedForward(
                        d_model, d_ff, dropout_rate),
                    dropout_rate=dropout_rate)),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab),
                                PositionalEncoding(d_model, dropout)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab),
                                PositionalEncoding(d_model, dropout)),
        generator=Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model