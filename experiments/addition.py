from re import X

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import addition_task

num_bits = 31  # TODO: support signed ints for full int32.
num_iters = 8
d_head = 8
num_heads = 12
d_ff = 256
dropout_rate = 0.0
batch_size = 64
learning_rate = 0.1

key = jax.random.PRNGKey(0)

data_key, weights_key, dropout_key = jax.random.split(key, 3)

model = addition_task.AdditionModel(num_heads=num_heads,
                                    num_iters=num_iters,
                                    d_head=d_head,
                                    d_ff=d_ff,
                                    dropout=nn.Dropout(rate=dropout_rate,
                                                       deterministic=True))
g = addition_task.data_generator(data_key,
                                 batch_size=batch_size,
                                 num_bits=num_bits)

init_xs, _ = next(g)
weights = model.init(key, init_xs[0, ...])

opt = optax.adam(learning_rate)
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