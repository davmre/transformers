import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn

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

        encoder = transformer_lib.Encoder(features=d_model,
                                          num_layers=num_layers,
                                          layer_fn=build_layer)

        x = jax.random.normal(jax.random.PRNGKey(42),
                              batch_shape + [n, d_model])

        rngs = {
            'params': jax.random.PRNGKey(0),
            'dropout': jax.random.PRNGKey(1)
        }
        weights = encoder.init(rngs, x)
        print("WEIHTS", weights)
        y = encoder.apply(weights, x, rngs=rngs)
        print("RESULT Y", y, y.shape)