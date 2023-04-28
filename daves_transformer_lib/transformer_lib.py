import functools
from typing import Callable, List

import jax
from jax import numpy as jnp

from flax import linen as nn


def causal_dependence(num_positions):
    #r = jnp.arange(num_positions)
    # rows index outputs, columns inputs
    return jnp.tril(jnp.ones([num_positions,
                              num_positions]))  #r[:, jnp.newaxis] >= r


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    features: int
    num_layers: int
    layer_fn: Callable

    def setup(self):
        self.layers = [self.layer_fn() for _ in range(self.num_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    size: int
    dropout: nn.Module

    def setup(self):
        self.norm = nn.LayerNorm()

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
            nn.Dense(self.d_model, kernel_init=nn.initializers.normal(0.02))
            for _ in range(4)
        ]

    def __call__(self, query, key, value, mask=None):
        # q, k, v all have shape [..., n, h * d_k].

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        @functools.partial(jnp.vectorize, signature='(n,m),(n,m),(n,m)->(n,m)')
        def do_attention(q, k, v):
            # Convert to shape [..., h, n, d_k].
            orig_shape = q.shape
            q, k, v = [
                jnp.transpose(jnp.reshape(t, [-1, self.h, self.d_k]), (1, 0, 2))
                for t in (q, k, v)
            ]
            x, _ = attention(q, k, v, mask=mask, dropout=self.dropout)
            # x shape [h, n, d_k] -> [n, h * d_k]
            return jnp.reshape(jnp.transpose(x, (1, 0, 2)), orig_shape)

        return self.final_linear(do_attention(query, key, value))


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

    def __call__(self, x):
        return self.dropout(self.w_2(jax.nn.gelu(self.w_1(x))))
