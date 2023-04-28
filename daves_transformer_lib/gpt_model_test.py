from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import gpt_model


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


class GPTModelTests(parameterized.TestCase):

    @parameterized.parameters([{'block_size': 1}, {'block_size': 100}])
    def test_generate_deterministic(self, block_size):
        key = jax.random.PRNGKey(0)
        xs = jnp.array([0])

        model = IncrementModel(vocab_size=4, block_size=block_size)
        weights = model.init(jax.random.PRNGKey(0), xs)
        tokens_with_lps = gpt_model.generate(key,
                                             model=model,
                                             weights=weights,
                                             context=xs,
                                             num_tokens=10)
        tokens = [t for (t, lps) in tokens_with_lps]
        self.assertSequenceEqual(tokens, [1, 2, 3, 0, 1, 2, 3, 0, 1, 2])

    def test_causal_masking(self):
        key = jax.random.PRNGKey(0)
        model = gpt_model.GPTModel(vocab_size=4,
                                   num_heads=2,
                                   num_layers=3,
                                   d_head=3,
                                   d_ff=7,
                                   block_size=16,
                                   dropout=nn.Dropout(rate=0.0,
                                                      deterministic=True))
        context1 = jnp.zeros([model.block_size], dtype=jnp.int32)
        context2 = jnp.ones([model.block_size], dtype=jnp.int32)
        context2 = context2.at[0].set(context1[0])

        weights = model.init(key, context1)

        logits1 = model.apply(weights, context1)
        logits2 = model.apply(weights, context2)
        self.assertSequenceAlmostEqual(logits1[0, :], logits2[0, :])

        t_with_lps1 = gpt_model.generate(key=key,
                                         model=model,
                                         weights=weights,
                                         context=context1,
                                         context_length=1,
                                         num_tokens=20)
        t_with_lps2 = gpt_model.generate(key=key,
                                         model=model,
                                         weights=weights,
                                         context=context2,
                                         context_length=1,
                                         num_tokens=20)

        for (t1, lp1s), (t2, lps2) in zip(t_with_lps1, t_with_lps2):
            self.assertEqual(int(t1), int(t2))
            self.assertSequenceAlmostEqual(lp1s, lps2)