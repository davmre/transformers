from enum import Enum
import functools
import math
from sys import float_info
from sys import int_info
from typing import Callable, List

from ml_collections import config_dict
import tree

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
from flax import struct

from daves_transformer_lib import transformer_lib


@functools.partial(jnp.vectorize, signature='(),(k)->()')
def log_loss(y, y_pred):
    # Allow a slight type mismatch: y_pred are unnormalized logits,
    # while `y` is an integer index.
    assert (y.shape == ())
    logits = jax.nn.log_softmax(y_pred)
    return -logits[y]


class GPTModel(nn.Module):
    vocab_size: int
    block_size: int
    num_heads: int
    num_layers: int
    d_head: int
    d_ff: int
    dropout: nn.Module
    internal_dtype = jnp.float32

    @staticmethod
    def get_default_config():
        config = config_dict.ConfigDict()
        config.vocab_size = 2
        config.num_layers = 6
        config.d_head = 32
        config.num_heads = 6
        config.d_ff = 768
        config.dropout_rate = 0.1
        return config

    def rngs(self, key):
        return {'dropout': key}

    @nn.compact
    def __call__(self, xs):
        d_model = self.d_head * self.num_heads

        # xs is an array of integer indices?
        num_positions = xs.shape[-1]
        token_embed = nn.Embed(num_embeddings=self.vocab_size,
                               features=d_model,
                               embedding_init=nn.initializers.normal(0.02))
        position_embed = nn.Embed(num_embeddings=self.block_size,
                                  features=d_model,
                                  embedding_init=nn.initializers.normal(0.02))
        zs = token_embed(xs) + position_embed(jnp.arange(num_positions))

        for _ in range(self.num_layers):
            transformer_layer = transformer_lib.EncoderLayer(
                size=d_model,
                self_attn=transformer_lib.MultiHeadedAttention(
                    self.num_heads, d_model, dropout=self.dropout),
                feed_forward=transformer_lib.PositionwiseFeedForward(
                    d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    w2_init_stddev=0.02 * math.sqrt(2 * self.num_layers)),
                dropout=self.dropout)
            zs = transformer_layer(zs,
                                   mask=transformer_lib.causal_dependence(
                                       num_positions=num_positions))

        zs = nn.LayerNorm()(zs)
        ys = nn.Dense(self.vocab_size,
                      use_bias=False,
                      dtype=self.internal_dtype,
                      kernel_init=nn.initializers.normal(0.02))(zs)
        return ys


def weight_decay_mask(params):

    def f(p, x):
        # Apply weight decay only to weight matrices of Dense layers.
        weight_decay = (p[-1] == 'kernel')
        return weight_decay

    return tree.map_structure_with_path(f, params)
