from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import generate
from daves_transformer_lib import transformer_lib


class IncrementModel(nn.Module):
    """Dummy model that puts (almost) all mass on `input + 1`."""
    vocab_size: int
    block_size: int

    def rngs(self, key):
        return {}

    @nn.compact
    def __call__(self, xs, kv_caches=None):
        desired_ys = (xs + 1) % self.vocab_size
        logits = 10 * jax.nn.one_hot(desired_ys, num_classes=self.vocab_size)
        return logits, kv_caches


class GenerateTests(parameterized.TestCase):

    @parameterized.parameters([{'block_size': 1}, {'block_size': 100}])
    def test_generate_deterministic(self, block_size):
        key = jax.random.PRNGKey(0)
        xs = jnp.array([0])

        model = IncrementModel(vocab_size=4, block_size=block_size)
        weights = model.init(jax.random.PRNGKey(0), xs)
        tokens = generate.generate(key,
                                   model=model,
                                   weights=weights,
                                   context=xs)
        self.assertSequenceEqual(list(tokens), [1, 2, 3, 0, 1, 2, 3, 0, 1, 2])

    @parameterized.parameters([
        {
            'num_layers': 1,
            'limit_to_context_length': False
        },
        {
            # For multilayer networks, the key-value cache *does* subtly change
            # the generation probabilities by implicitly extending the context
            # length: activations at the beginning of the cache will themselves
            # depend on older activations outside of the current context. So we
            # test equality only up to the context length.
            'num_layers': 2,
            'limit_to_context_length': True
        }
    ])
    def test_kv_cache_doesnt_change_generation(self, num_layers,
                                               limit_to_context_length):
        key = jax.random.PRNGKey(0)
        xs = jnp.array([0])

        model = transformer_lib.GPTModel(
            vocab_size=4,
            num_heads=1,
            num_layers=num_layers,
            d_model=3,
            d_ff=6,
            block_size=5,
            dropout=nn.Dropout(rate=0., deterministic=True),
            # TODO: position embeddings shouldn't
            # affect how the cache works.
            use_position_embeddings=False)
        weights = model.init(jax.random.PRNGKey(0), xs)
        tokens_with_lps_no_cache = generate.generate(key,
                                                     model=model,
                                                     weights=weights,
                                                     context=xs,
                                                     include_logprobs=True)

        tokens_with_lps = generate.generate_with_kv_cache(key,
                                                          model=model,
                                                          weights=weights,
                                                          context=xs,
                                                          include_logprobs=True)

        for i in range(model.block_size +
                       (0 if limit_to_context_length else 10)):
            t_nc, lp_nc = next(tokens_with_lps_no_cache)
            t, lp = next(tokens_with_lps)
            self.assertEqual(t, t_nc)
            # FAILS! Positional encoding handling would cause this but
            # even fails when that's commented out. So wtf?
            np.testing.assert_array_almost_equal(lp, lp_nc)
