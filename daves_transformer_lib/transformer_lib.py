import functools
import math
from typing import Callable, Optional

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

import jax
from jax import numpy as jnp

from flax import linen as nn


@functools.partial(jnp.vectorize, signature='(),(k)->()')
def log_loss(y: Int[Array, "..."],
             y_pred: Float[Array, "... k"]) -> Float[Array, "..."]:
    # Allow a slight type mismatch: y_pred are unnormalized logits,
    # while `y` is an integer index.
    assert (y.shape == ())
    logits = jax.nn.log_softmax(y_pred)
    return -logits[y]


def causal_dependence(
        num_positions: int) -> Float[Array, "num_positions num_positions"]:
    #r = jnp.arange(num_positions)
    # rows index outputs, columns inputs
    return jnp.tril(jnp.ones([num_positions, num_positions]))


class MultiHeadedAttention(nn.Module):

    h: int
    d_model: int
    dropout: nn.Module

    def setup(self):
        self.d_k = self.d_model // self.h
        self.qkv_linear = nn.Dense(self.d_model * 3,
                                   kernel_init=nn.initializers.normal(0.02))
        self.final_linear = nn.Dense(self.d_model,
                                     kernel_init=nn.initializers.normal(0.02))

    def __call__(
        self,
        x: Float[Array, "... num_positions d_model"],
        mask: Optional[Float[Array, "num_positions num_positions"]] = None
    ) -> Float[Array, "... num_positions d_model"]:
        input_shape = x.shape

        qkv = self.qkv_linear(x)
        # Split into individual attention heads.
        qkv = jnp.reshape(qkv, input_shape[:-1] + (self.h, self.d_k * 3))

        # Convert shape [..., n, h * d_k] to shape [..., h, n, d_k].
        qkv = jnp.swapaxes(qkv, -3, -2)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(
            float(self.d_k))
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        p_attn = jax.nn.softmax(scores, axis=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = jnp.matmul(p_attn, value)

        # x shape [h, n, d_k] -> [n, h * d_k]
        x = jnp.reshape(jnp.swapaxes(x, -3, -2), input_shape)
        return self.final_linear(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    d_model: int
    d_ff: int
    dropout: nn.Module
    w1_init_stddev: float = 0.02
    w2_init_stddev: float = 0.02

    def setup(self):
        self.w_1 = nn.Dense(self.d_ff,
                            kernel_init=nn.initializers.normal(
                                self.w1_init_stddev))
        self.w_2 = nn.Dense(self.d_model,
                            kernel_init=nn.initializers.normal(
                                self.w2_init_stddev))

    def __call__(self, x: Float[Array,
                                "... d_model"]) -> Float[Array, "... d_model"]:
        return self.dropout(self.w_2(jax.nn.gelu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    size: int
    dropout: nn.Module

    def setup(self):
        self.norm = nn.LayerNorm()

    def __call__(self, x: Float[Array, "... d_model"],
                 sublayer: Callable) -> Float[Array, "... d_model"]:
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
        x = self.attn_sublayer(x, sublayer=lambda x: self.self_attn(x, mask))
        return self.feed_forward_sublayer(x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    num_layers: int
    layer_fn: Callable

    def setup(self):
        self.layers = [self.layer_fn() for _ in range(self.num_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class GPTModel(nn.Module):
    vocab_size: int
    block_size: int
    num_heads: int
    num_layers: int
    d_head: int
    d_ff: int
    dropout: nn.Module
    internal_dtype = jnp.float32

    def rngs(self, key: jax.random.KeyArray):
        return {'dropout': key}

    @nn.compact
    def __call__(
        self, xs: Int[Array, "... num_positions"]
    ) -> Float[Array, "... num_positions vocab_size"]:
        d_model = self.d_head * self.num_heads

        # `xs`` is an array of integer indices.
        num_positions = xs.shape[-1]
        token_embed = nn.Embed(num_embeddings=self.vocab_size,
                               features=d_model,
                               embedding_init=nn.initializers.normal(0.02))
        position_embed = nn.Embed(num_embeddings=self.block_size,
                                  features=d_model,
                                  embedding_init=nn.initializers.normal(0.02))
        zs = token_embed(xs) + position_embed(jnp.arange(num_positions))

        for _ in range(self.num_layers):
            transformer_layer = EncoderLayer(
                size=d_model,
                self_attn=MultiHeadedAttention(self.num_heads,
                                               d_model,
                                               dropout=self.dropout),
                feed_forward=PositionwiseFeedForward(
                    d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    w2_init_stddev=0.02 * math.sqrt(2 * self.num_layers)),
                dropout=self.dropout)
            zs = transformer_layer(
                zs, mask=causal_dependence(num_positions=num_positions))

        zs = nn.LayerNorm()(zs)
        ys = nn.Dense(self.vocab_size,
                      use_bias=False,
                      dtype=self.internal_dtype,
                      kernel_init=nn.initializers.normal(0.02))(zs)
        return ys
