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
        return jax.nn.log_softmax(self.proj(x), axis=-1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    features: int
    num_layers: int
    layer_fn: Callable

    def setup(self):
        self.layers = [self.layer_fn() for _ in range(self.num_layers)]
        self.norm = LayerNorm(self.features)

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    features: int
    eps: float = 1e-6

    def setup(self):
        self.a_2 = self.param('a_2', lambda k, s: jnp.ones(s), (self.features,))
        self.b_2 = self.param('b_2', lambda k, s: jnp.zeros(s),
                              (self.features,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    size: int
    dropout: nn.Module

    def setup(self):
        self.norm = LayerNorm(self.size)

    def __call__(self, x, sublayer: Callable):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    size: int
    self_attn: nn.Module
    feed_forward: nn.Module
    dropout: nn.Module

    def setup(self):
        self.attn_sublayer = SublayerConnection(self.size, dropout=self.dropout)
        self.feed_forward_sublayer = SublayerConnection(self.size,
                                                        dropout=self.dropout)

    def __call__(self, x, mask):
        print('encoder layer with x', x.shape)
        x = self.attn_sublayer(x,
                               sublayer=lambda x: self.self_attn(x, x, x, mask))
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
    dropout: nn.Module

    def setup(self):
        self.d_k = self.d_model // self.h
        [
            self.query_linear, self.key_linear, self.value_linear,
            self.final_linear
        ] = [
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform())
            for _ in range(4)
        ]

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
            jnp.transpose(jnp.reshape(t, [-1, self.h, self.d_k]), (1, 0, 2))
            for t in (query, key, value)
        ]

        x, _ = attention(query, key, value, dropout=self.dropout)
        # x shape [h, n, d_k] -> [n, h * d_k]
        x = jnp.reshape(jnp.transpose(x, (1, 0, 2)), (-1, self.h * self.d_k))
        return self.final_linear(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    d_model: int
    d_ff: int
    dropout: nn.Module

    def setup(self):
        self.w_1 = nn.Dense(self.d_ff,
                            kernel_init=nn.initializers.xavier_uniform())
        self.w_2 = nn.Dense(self.d_model,
                            kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, x):
        return self.w_2(self.dropout(jax.nn.relu(self.w_1(x))))
