import dataclasses
import typing

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn

from daves_transformer_lib import generate
from daves_transformer_lib import transformer_lib


class KeyValueCacheTests(parameterized.TestCase):

    def test_cache_mechanics(self):

        context_size = 3
        d_model = 1
        seed = jax.random.PRNGKey(42)
        seed1, seed2 = jax.random.split(seed)
        # Initialize an empty cache
        kv_cache = transformer_lib.KeyValueCache(cache=jnp.zeros(
            [context_size, d_model * 2]),
                                                 valid_size=0)

        # Update 1: add to the cache but do not fill it.
        num_new_positions = context_size - 2
        kvs1 = jax.random.normal(seed1, [num_new_positions, d_model * 2])
        mask = jnp.tri(num_new_positions)
        updated_cache, retrieved_kvs, retrieved_mask = kv_cache.update(
            kvs1, mask=mask)

        self.assertEqual(int(updated_cache.valid_size), num_new_positions)

        # Retrieved KVs should match passed-in KVs, padded up to context size.
        self.assertEqual(retrieved_kvs.shape[-2], context_size)
        np.testing.assert_array_almost_equal(retrieved_kvs[-num_new_positions:],
                                             kvs1)
        # Retrieved mask should specialize to the original mask on passed-in KVs.
        self.assertSequenceEqual(retrieved_mask.shape,
                                 [num_new_positions, context_size])
        np.testing.assert_array_almost_equal(
            retrieved_mask[:, -num_new_positions:], mask)
        # No attention to uninitialized cache region.
        np.testing.assert_array_almost_equal(
            retrieved_mask[:, :-num_new_positions], 0.)

        # Update 2: overfill the cache
        num_new_positions = context_size - 1
        kvs2 = jax.random.normal(seed2, [num_new_positions, d_model * 2])
        mask = jnp.tri(num_new_positions)
        updated_cache2, retrieved_kvs2, retrieved_mask2 = updated_cache.update(
            kvs2, mask=mask)

        self.assertEqual(int(updated_cache2.valid_size), context_size)

        # Updated KVs should match passed-in KVs, padded up to context size.
        self.assertEqual(retrieved_kvs2.shape[-2], context_size)
        np.testing.assert_array_almost_equal(
            retrieved_kvs2[-num_new_positions:], kvs2)
        # Updated mask should specialize to the original mask on passed-in KVs.
        self.assertSequenceEqual(retrieved_mask2.shape,
                                 [num_new_positions, context_size])
        np.testing.assert_array_almost_equal(
            retrieved_mask2[:, -num_new_positions:], mask)
        # All new positions attend to all retrieved positions.
        np.testing.assert_array_almost_equal(
            retrieved_mask2[:, :num_new_positions],
            jnp.ones([num_new_positions, num_new_positions]))


class MixtureOfExpertsTests(parameterized.TestCase):

    @parameterized.parameters([
        {
            'num_active_experts': 1
        },
        {
            'num_active_experts': 2
        },
    ])
    def test_mixture_of_dense(self, num_active_experts):
        num_experts = 8
        d_in = 3
        d_out = 2
        mixture_layer = transformer_lib.NaiveSparseMixtureOfExpertsLayer(
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            sublayer=nn.Dense,
            sublayer_args=(d_out,))

        x = jnp.ones([d_in])
        rngs = {
            'params': jax.random.PRNGKey(0),
            'expert_choice': jax.random.PRNGKey(1)
        }
        weights = mixture_layer.init(rngs, x)
        print("GOT WEIGHTS", weights)
        #with jax.checking_leaks():
        y = mixture_layer.apply(weights, x, rngs=rngs)
        print("GOT Y", y)


class TransformerTests(parameterized.TestCase):

    @parameterized.parameters([
        {
            'batch_shape': []
        },
        {
            'batch_shape': [3]
        },
    ])
    def test_build_layer_stack(self, batch_shape):
        n = 7
        h = 2
        d_k = 8
        d_model = h * d_k
        d_ff = 32
        num_layers = 2
        dropout_rate = 0.1

        dropout = nn.Dropout(dropout_rate, deterministic=False)

        layer = transformer_lib.TransformerBlock(
            size=d_model,
            self_attn=transformer_lib.MultiHeadedAttention(h,
                                                           d_model,
                                                           dropout=dropout),
            feed_forward=transformer_lib.PositionwiseFeedForward(d_model, d_ff),
            dropout=dropout)

        x = jax.random.normal(jax.random.PRNGKey(42),
                              batch_shape + [n, d_model])

        rngs = {
            'params': jax.random.PRNGKey(0),
            'dropout': jax.random.PRNGKey(1)
        }
        weights = layer.init(rngs, x)
        x, _ = layer.apply(weights, x, rngs=rngs)
        x, _ = layer.apply(weights, x, rngs=rngs)

    def test_causal_masking(self):
        key = jax.random.PRNGKey(0)
        model = transformer_lib.GPTModel(vocab_size=4,
                                         num_heads=2,
                                         num_layers=1,
                                         d_model=6,
                                         d_ff=7,
                                         block_size=16,
                                         dropout=nn.Dropout(rate=0.0,
                                                            deterministic=True))
        context1 = jnp.zeros([model.block_size], dtype=jnp.int32)
        context2 = jnp.ones([model.block_size], dtype=jnp.int32)
        context2 = context2.at[0].set(context1[0])

        weights = model.init(key, context1)

        logits1, _ = model.apply(weights, context1)
        logits2, _ = model.apply(weights, context2)
        logits1 = typing.cast(jax.Array, logits1)
        logits2 = typing.cast(jax.Array, logits2)
        self.assertSequenceAlmostEqual(logits1[0, :], logits2[0, :])

        t_with_lps1 = generate.generate(key=key,
                                        model=model,
                                        weights=weights,
                                        context=context1,
                                        context_length=1,
                                        include_logprobs=True)
        t_with_lps2 = generate.generate(key=key,
                                        model=model,
                                        weights=weights,
                                        context=context2,
                                        context_length=1,
                                        include_logprobs=True)

        for (t1, lp1s), (t2, lps2), _ in zip(t_with_lps1, t_with_lps2,
                                             range(20)):
            self.assertEqual(int(t1), int(t2))
            self.assertSequenceAlmostEqual(lp1s, lps2)