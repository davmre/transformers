import dataclasses
import functools
import math
from typing import Callable, Dict, List, Optional, Tuple, Type

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax import struct


@struct.dataclass
class KeyValueCache:
    cache: Float[Array, "num_cached_positions d_kv"]
    valid_size: Int

    @property
    def cache_size(self):
        return self.cache.shape[-2]

    def update(
        self, key_value: Float[Array, "num_new_positions twice_d_model"],
        mask: Optional[Float[Array, "num_new_positions num_new_positions"]]
    ) -> Tuple["KeyValueCache", Float[Array, "num_positions twice_d_model"],
               Float[Array, "num_new_positions num_positions"]]:
        num_new_positions = key_value.shape[-2]
        assert (num_new_positions < self.cache_size)

        new_valid_size = jnp.minimum(self.cache_size,
                                     self.valid_size + num_new_positions)

        # Add the cached positions.
        key_value = jnp.concatenate([self.cache[num_new_positions:], key_value],
                                    axis=-2)

        # Augment the mask to attend only to positions where the cache
        # is valid. The augmented mask has dimension
        # `[num_new_positions, cache_size]`.
        if mask is None:
            # Create a dummy mask if needed.
            mask = jnp.ones([num_new_positions, num_new_positions],
                            dtype=self.cache.dtype)

        # cache size 5
        # valid size 1
        # new positions 1
        # we want a 4-sequence in which final 2 entries are 1
        cache_validity_indicators = jnp.where(
            jnp.arange(self.cache_size - num_new_positions, 0, -1) <=
            self.valid_size, 1., 0.)
        cache_validity_mask = cache_validity_indicators[
            jnp.newaxis, :] * jnp.ones([num_new_positions, 1])
        mask = jnp.concatenate([cache_validity_mask, mask], axis=-1)

        # Update the cache for the next step.
        new_cache = KeyValueCache(cache=key_value, valid_size=new_valid_size)

        return new_cache, key_value, mask


def causal_dependence(
        num_positions: int) -> Float[Array, "num_positions num_positions"]:
    #r = jnp.arange(num_positions)
    # rows index outputs, columns inputs
    return jnp.tril(jnp.ones([num_positions, num_positions]))


class NaiveSparseMixtureOfExpertsLayer(nn.Module):
    # it'd be nice to have a generic wrapper that, for any layer,
    # instantiates N copies of it and then chooses one/two to run.
    # this doesn't allow for expert choice routing but it does
    # make things generally nice.
    # I think first I write a MoE for a general dense layer
    # then maybe make it meta as a wrapper for attention etc.

    num_experts: int
    num_active_experts: int
    sublayer: Type[nn.Module]
    sublayer_args: Tuple = ()
    sublayer_kwargs: Dict = struct.field(default_factory=dict)

    @nn.compact
    def __call__(self, x):
        # Sample experts by adding Gaussian noise to the logits, where the noise
        # itself has data-dependent scale. This is the formulation from the
        # original Shazeer et al. (2017) paper (Outrageously Large Neural
        # Networks: The Sparsely-Gated Mixture-of-Experts Layer,
        # https://arxiv.org/abs/1701.06538 sec 2.1)
        print("calling with x", x.shape)
        logits = nn.Dense(self.num_experts, name='experts_linear')(x)
        jitter_scales = nn.softplus(
            nn.Dense(self.num_experts, name='experts_jitter_scales_linear')(x))
        jitter = jitter_scales * jax.random.normal(
            key=self.make_rng('expert_choice'), shape=jitter_scales.shape)

        # Select only the top experts and return a weighted sum of their
        # contributions.
        # TODO(davmre): investigate speed of top_k vs argmax (cf https://github.com/google/jax/issues/9940)
        selected_logits, selected_expert_idxs = jax.lax.top_k(
            logits + jitter, self.num_active_experts)
        sparse_logits = jnp.zeros_like(logits).at[selected_expert_idxs].set(
            selected_logits)
        gate_probs = nn.softmax(sparse_logits)

        # Run the input through all experts in parallel, then sum the results according
        # to the (sparse) gating weights. This throws away the computational
        # benefits of a sparse mixture, but keeps the code simple and allows
        # us to guarantee that each input gets its choice of expert. Given the
        # constraint of fixed-shape arrays, the alternative would be for each
        # expert to process a bounded number of inputs in each batch
        # (expert-choice routing), which is computationally more efficient but
        # will sometimes route inputs incorrectly. That choice makes sense at
        # scale, but for research purposes we prefer to trade the computational
        # benefits of sparsity in exchange for reliable routing.
        batch_of_experts = nn.vmap(
            self.sublayer,
            axis_size=self.num_experts,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None  # type: ignore
        )(*self.sublayer_args, **self.sublayer_kwargs)
        results = batch_of_experts(x)
        return jnp.sum(gate_probs[..., jnp.newaxis] * results, axis=0)


class MultiHeadedAttention(nn.Module):

    h: int
    d_model: int
    dropout: nn.Module

    def setup(self):
        self.d_k = self.d_model // self.h
        self.q_linear = nn.Dense(self.d_model,
                                 kernel_init=nn.initializers.normal(0.02))
        self.kv_linear = nn.Dense(self.d_model * 2,
                                  kernel_init=nn.initializers.normal(0.02))
        self.final_linear = nn.Dense(self.d_model,
                                     kernel_init=nn.initializers.normal(0.02))

    def split_heads(self, x):
        return jnp.reshape(x, x.shape[:-1] + (self.h, -1))

    def merge_heads(self, x):
        return jnp.reshape(x, x.shape[:-2] + (-1,))

    def __call__(
        self,
        x: Float[Array, "... num_positions d_model"],
        mask: Optional[Float[Array, "num_positions num_positions"]] = None,
        kv_cache: Optional[KeyValueCache] = None
    ) -> Tuple[Float[Array, "... num_positions d_model"],
               Optional[KeyValueCache]]:
        num_input_positions = x.shape[-2]

        query = self.q_linear(x)
        key_value = self.kv_linear(x)
        if kv_cache is not None:
            kv_cache, key_value, mask = kv_cache.update(key_value, mask=mask)

        key, value = jnp.split(key_value, 2, axis=-1)

        # Split into individual attention heads.
        query, key, value = (self.split_heads(v) for v in (query, key, value))

        # Convert shape [..., n, h, d_k] to shape [..., h, n, d_k].
        query = jnp.swapaxes(query, -3, -2)
        key = jnp.swapaxes(key, -3, -2)
        value = jnp.swapaxes(value, -3, -2)

        scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(
            float(self.d_k))
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)
        p_attn = jax.nn.softmax(scores, axis=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = jnp.matmul(p_attn, value)

        # x shape [h, n, d_k] -> [n, h * d_k]
        x = self.merge_heads(jnp.swapaxes(x, -3, -2))
        x = self.final_linear(x)
        return x, kv_cache


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    d_model: int
    d_ff: int
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
        return self.w_2(jax.nn.gelu(self.w_1(x)))


class TransformerBlock(nn.Module):
    size: int
    self_attn: nn.Module
    feed_forward: nn.Module
    dropout: nn.Module

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None, kv_cache=None):
        # Attention block
        x_attn = self.norm1(x)
        x_attn, kv_cache = self.self_attn(x_attn, mask=mask, kv_cache=kv_cache)
        x = x + self.dropout(x_attn)

        # Feedforward block
        x_ff = self.norm2(x)
        x_ff = self.feed_forward(x_ff)
        x = x + self.dropout(x_ff)

        return x, kv_cache


class GPTModel(nn.Module):
    vocab_size: int
    block_size: int
    num_heads: int
    num_layers: int
    d_model: int
    d_ff: int
    dropout: nn.Module
    use_position_embeddings: bool = True
    internal_dtype = jnp.float32
    num_ff_experts: int = 1
    num_ff_experts_active: int = 1

    def rngs(self, key: jax.random.KeyArray):
        dropout_key, expert_choice_key = jax.random.split(key)
        rngs = {'dropout': dropout_key, 'expert_choice': expert_choice_key}
        return rngs

    def initialize_kv_caches(self) -> List[KeyValueCache]:
        return [
            KeyValueCache(cache=jnp.zeros([self.block_size, self.d_model * 2],
                                          dtype=self.internal_dtype),
                          valid_size=0) for _ in range(self.num_layers)
        ]

    @nn.compact
    def __call__(
        self,
        xs: Int[Array, "... num_positions"],
        kv_caches: Optional[List[KeyValueCache]] = None
    ) -> Tuple[Float[Array, "... num_positions vocab_size"],
               Optional[List[KeyValueCache]]]:

        # `xs`` is an array of integer indices.
        num_positions = xs.shape[-1]
        token_embed = nn.Embed(num_embeddings=self.vocab_size,
                               features=self.d_model,
                               embedding_init=nn.initializers.normal(0.02))
        zs = token_embed(xs)
        if self.use_position_embeddings:
            position_embed = nn.Embed(
                num_embeddings=self.block_size,
                features=self.d_model,
                embedding_init=nn.initializers.normal(0.02))
            zs += position_embed(jnp.arange(num_positions))

        for layer_idx in range(self.num_layers):

            ff_layer_type = PositionwiseFeedForward
            ff_args = (self.d_model, self.d_ff)
            ff_kwargs = dict(w2_init_stddev=0.02 *
                             math.sqrt(2 * self.num_layers))
            if self.num_ff_experts > 1:
                ff_kwargs = dict(num_experts=self.num_ff_experts,
                                 num_active_experts=self.num_ff_experts_active,
                                 sublayer=ff_layer_type,
                                 sublayer_args=ff_args,
                                 sublayer_kwargs=ff_kwargs)
                ff_args = ()
                ff_layer_type = NaiveSparseMixtureOfExpertsLayer

            ff_layer = nn.vmap(  # Parallelize over sequence positions.
                ff_layer_type,
                variable_axes={'params': None},
                split_rngs={
                    'params': False,
                    'expert_choice': True
                },
                in_axes=0)(*ff_args, **ff_kwargs)  # type: ignore
            attn_layer = MultiHeadedAttention(self.num_heads,
                                              self.d_model,
                                              dropout=self.dropout)
            transformer_layer = TransformerBlock(size=self.d_model,
                                                 self_attn=attn_layer,
                                                 feed_forward=ff_layer,
                                                 dropout=self.dropout)

            zs, updated_kv_cache = transformer_layer(
                zs,
                mask=causal_dependence(num_positions=num_positions),
                kv_cache=kv_caches[layer_idx] if kv_caches else None)
            if kv_caches:
                kv_caches[layer_idx] = updated_kv_cache

        zs = nn.LayerNorm()(zs)
        ys = nn.Dense(self.vocab_size,
                      use_bias=False,
                      dtype=self.internal_dtype,
                      kernel_init=nn.initializers.normal(0.02))(zs)
        return ys, kv_caches
