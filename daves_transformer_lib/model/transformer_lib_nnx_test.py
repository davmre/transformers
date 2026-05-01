import dataclasses
import typing

import jax
import numpy as np
from absl.testing import absltest, parameterized
from flax import nnx
from jax import numpy as jnp

from daves_transformer_lib import transformer_lib_nnx


class TransformerTests(parameterized.TestCase):
    @parameterized.parameters(
        [
            {"batch_shape": []},
            {"batch_shape": [3]},
        ]
    )
    def test_build_layer_stack(self, batch_shape):
        n = 7
        h = 2
        d_k = 8
        d_model = h * d_k
        d_ff = 32
        num_layers = 2

        rngs = nnx.Rngs(0)
        layer = transformer_lib_nnx.TransformerBlock(
            d_model=d_model,
            num_heads=h,
            d_ff=d_ff,
            rope=None,
            kernel_init=nnx.initializers.glorot_normal(),
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.PRNGKey(42), batch_shape + [n, d_model])

        x = layer(x)
        x = layer(x)

    def test_causal_masking(self):
        key = jax.random.PRNGKey(0)

        rngs = nnx.Rngs(0)

        model = transformer_lib_nnx.TransformerLanguageModel(
            vocab_size=4,
            num_heads=2,
            num_layers=1,
            d_model=8,
            d_ff=7,
            context_length=16,
            rngs=rngs,
        )
        context1 = jnp.zeros([model._context_length], dtype=jnp.int32)
        context2 = jnp.ones([model._context_length], dtype=jnp.int32)
        context2 = context2.at[0].set(context1[0])

        logits1 = model(context1)
        logits2 = model(context2)
        self.assertSequenceAlmostEqual(logits1[0, :], logits2[0, :])
