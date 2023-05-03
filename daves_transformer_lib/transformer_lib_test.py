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


class TransformerTests(parameterized.TestCase):

    @parameterized.parameters([
        {
            'batch_shape': []
        },
        {
            'batch_shape': [3]
        },
    ])
    def test_build_encoder(self, batch_shape):
        n = 7
        h = 2
        d_k = 8
        d_model = h * d_k
        d_ff = 32
        num_layers = 2
        dropout_rate = 0.1

        def build_layer():
            dropout = nn.Dropout(dropout_rate, deterministic=False)

            return transformer_lib.EncoderLayer(
                size=d_model,
                self_attn=transformer_lib.MultiHeadedAttention(h,
                                                               d_model,
                                                               dropout=dropout),
                feed_forward=transformer_lib.PositionwiseFeedForward(
                    d_model, d_ff, dropout=dropout),
                dropout=dropout)

        encoder = transformer_lib.Encoder(num_layers=num_layers,
                                          layer_fn=build_layer)

        x = jax.random.normal(jax.random.PRNGKey(42),
                              batch_shape + [n, d_model])

        rngs = {
            'params': jax.random.PRNGKey(0),
            'dropout': jax.random.PRNGKey(1)
        }
        weights = encoder.init(rngs, x)
        y = encoder.apply(weights, x, rngs=rngs)

    def test_causal_masking(self):
        key = jax.random.PRNGKey(0)
        model = transformer_lib.GPTModel(vocab_size=4,
                                         num_heads=2,
                                         num_layers=1,
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
        logits1 = typing.cast(jax.Array, logits1)
        logits2 = typing.cast(jax.Array, logits2)
        self.assertSequenceAlmostEqual(logits1[0, :], logits2[0, :])

        t_with_lps1 = generate.generate(key=key,
                                        model=model,
                                        weights=weights,
                                        context=context1,
                                        context_length=1,
                                        num_tokens=20,
                                        include_logprobs=True)
        t_with_lps2 = generate.generate(key=key,
                                        model=model,
                                        weights=weights,
                                        context=context2,
                                        context_length=1,
                                        num_tokens=20,
                                        include_logprobs=True)

        for (t1, lp1s), (t2, lps2) in zip(t_with_lps1, t_with_lps2):
            self.assertEqual(int(t1), int(t2))
            self.assertSequenceAlmostEqual(lp1s, lps2)