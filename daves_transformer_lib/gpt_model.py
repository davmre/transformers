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


@functools.partial(jax.jit, static_argnums=(0,))
def generate_step(model,
                  key,
                  weights,
                  context,
                  context_length,
                  top_k=None,
                  temperature=1.0):
    forward_key, sample_key = jax.random.split(key)
    all_logits = model.apply(weights, context, rngs=model.rngs(forward_key))
    logits = all_logits[..., context_length - 1, :] / temperature
    if top_k is not None:
        # TODO: find a more efficient top_k implementation.
        v = jnp.sort(logits, axis=-1)[..., -top_k]
        logits = jnp.where(logits < v[..., jnp.newaxis], -jnp.inf, logits)
    probs = jax.nn.softmax(logits)
    c = jax.random.choice(sample_key, logits.shape[-1], shape=(), p=probs)
    return c, jax.nn.log_softmax(logits)


def generate(key,
             model,
             weights,
             context,
             num_tokens,
             context_length=None,
             top_k=None,
             temperature=1.0):

    # Pad context out to block size.
    if not context_length:
        context_length = len(context)
    assert (context_length > 0)
    context = jnp.concatenate([
        context,
        jnp.zeros([model.block_size - len(context)], dtype=context.dtype)
    ])

    for t in range(num_tokens):
        gen_key, key = jax.random.split(key)
        c, log_probs = generate_step(model,
                                     key=gen_key,
                                     weights=weights,
                                     context=context,
                                     context_length=context_length,
                                     top_k=top_k,
                                     temperature=temperature)
        yield c, log_probs

        # Add the generated character to the context for the next step.
        context_length = min(context_length + 1, model.block_size)
        if context_length < model.block_size:
            context = context.at[context_length - 1].set(c)
        else:
            context = jnp.concatenate(
                [context[1:], jnp.array([c], dtype=context.dtype)])
    return context