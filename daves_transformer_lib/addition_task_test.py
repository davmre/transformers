from re import X

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import addition_task


class AdditionTests(parameterized.TestCase):

    @parameterized.parameters([{
        'n': 0,
        'bits': [0]
    }, {
        'n': 1,
        'bits': [1]
    }, {
        'n': 2,
        'bits': [1, 0]
    }, {
        'n': 5,
        'bits': [1, 0, 1]
    }, {
        'n': 31,
        'bits': [1, 1, 1, 1, 1]
    }, {
        'n': 31,
        'bits': [0, 0, 0, 1, 1, 1, 1, 1]
    }, {
        'n': 1337,
        'bits': [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    }])
    def test_binary_encoding(self, n, bits):
        num_bits = len(bits)
        x = addition_task.int_to_binary_encoding(n, num_bits=num_bits)
        self.assertEqual(x.shape, (num_bits,))
        np.testing.assert_array_equal(x, jnp.asarray(bits))
        nn = addition_task.binary_encoding_to_int(x)
        self.assertEqual(n, nn)

    def test_data_generator(self):
        key = jax.random.PRNGKey(0)
        g = addition_task.data_generator(key, batch_size=1, num_bits=16)
        for _ in range(10):
            xs, y = next(g)
            xs, y = xs[0, ...], y[0, ...]  # Strip batch dim.

            # Manually compute bitwise sum including carry bits.
            z = xs[:, 0] + xs[:, 1]
            while np.any(z > 1):
                carry = np.where(z > 1, 1, 0)
                z = np.where(z > 1, 0, z) + jnp.concatenate(
                    [carry[1:], jnp.asarray([0])])
            np.testing.assert_array_equal(z, y)

    def test_addition_model_runs(self):
        model = addition_task.AdditionModel(num_heads=2,
                                            num_iters=3,
                                            d_head=3,
                                            d_ff=13,
                                            dropout=nn.Dropout(
                                                rate=0., deterministic=True))
        key = jax.random.PRNGKey(0)
        g = addition_task.data_generator(key, batch_size=1, num_bits=16)
        xs, y = next(g)

        weights = model.init(key, xs)
        xxs = model.apply(weights, xs)[..., 0]
        self.assertEqual(xxs.shape, y.shape)

    def test_addition_model_trains(self):
        model = addition_task.AdditionModel(num_heads=2,
                                            num_iters=3,
                                            d_head=3,
                                            d_ff=13,
                                            dropout=nn.Dropout(
                                                rate=0., deterministic=True))
        key = jax.random.PRNGKey(0)
        g = addition_task.data_generator(key, batch_size=1, num_bits=16)

        init_xs, _ = next(g)
        weights = model.init(key, init_xs)

        opt = optax.adam(0.1)
        opt_state = opt.init(weights)

        def loss_fn(weights):
            xs, y = next(g)
            xxs = model.apply(weights, xs)
            return jnp.mean(jnp.sum((y - xxs[..., 0])**2, axis=-1), axis=0)

        @jax.jit
        def train_step(weights, opt_state):
            loss, grad = jax.value_and_grad(loss_fn)(weights)
            updates, opt_state = opt.update(grad, opt_state)
            weights = optax.apply_updates(weights, updates)
            return loss, weights, opt_state

        for step in range(100):
            loss, weights, opt_state = train_step(weights, opt_state)
            print(f"step {step} loss {loss}")