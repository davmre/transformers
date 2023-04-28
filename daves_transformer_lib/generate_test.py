from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import generate


class IncrementModel(nn.Module):
    """Dummy model that puts (almost) all mass on `input + 1`."""
    vocab_size: int
    block_size: int

    def rngs(self, key):
        return {}

    @nn.compact
    def __call__(self, xs):
        desired_ys = (xs + 1) % self.vocab_size
        logits = 10 * jax.nn.one_hot(desired_ys, num_classes=self.vocab_size)
        return logits


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
                                   context=xs,
                                   num_tokens=10)
        self.assertSequenceEqual(list(tokens), [1, 2, 3, 0, 1, 2, 3, 0, 1, 2])
