import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from daves_transformer_lib import transformer_lib

class AgentLibTests(test_util.TestCase):

    def test_build_encoder(self):
        n = 7
        h = 2
        d_k = 8
        d_model = h * d_k
        d_ff = 32
        num_layers = 2
        dropout_rate = 0.1
        
        def build_layer():
            return transformer_lib.EncoderLayer(
                size=d_model,
                self_attn=transformer_lib.MultiHeadedAttention(h, d_model),
                feed_forward=transformer_lib.PositionwiseFeedForward(
                    d_model, d_ff, dropout_rate),
                dropout_rate=dropout_rate)
        
        encoder = transformer_lib.Encoder(
            features=d_model,
            num_layers=num_layers,
            layer_fn=build_layer)
        
        x = jnp.random.normal(jax.random.PRNGKey(42), [n, d_model])
        
        weights = encoder.init(x)
        print("WEIHTS", weights)