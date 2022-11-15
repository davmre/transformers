from re import X

from absl import app
from absl import flags
from ml_collections import config_dict
from ml_collections import config_flags
import tensorboardX

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
from flax.training import train_state
import optax

from daves_transformer_lib import addition_task

# Example invocation:
# `python3 experiments/addition.py --my_config.num_train_steps=4``
config = config_dict.ConfigDict()
config.num_bits = 31  # TODO: support signed ints for full int32.
config.num_iters = 8
config.d_head = 8
config.num_heads = 12
config.d_ff = 256
config.dropout_rate = 0.0
config.batch_size = 64
config.learning_rate = 1e-4
config.num_train_steps = 6
config.logdir = '/tmp/addition_logs'
_CONFIG = config_flags.DEFINE_config_dict('my_config', config)


def init_train_state(key, model, init_xs, config):
    weights = model.init(key, init_xs[0, ...])
    opt = optax.adam(config.learning_rate)
    return train_state.TrainState(step=0,
                                  apply_fn=model.apply,
                                  params=weights,
                                  tx=opt,
                                  opt_state=opt.init(weights))


def main(_):
    config = _CONFIG.value
    key = jax.random.PRNGKey(0)

    data_key, weights_key, dropout_key = jax.random.split(key, 3)

    model = addition_task.AdditionModel(num_heads=config.num_heads,
                                        num_iters=config.num_iters,
                                        d_head=config.d_head,
                                        d_ff=config.d_ff,
                                        dropout=nn.Dropout(
                                            rate=config.dropout_rate,
                                            deterministic=True))
    g = addition_task.data_generator(data_key,
                                     batch_size=config.batch_size,
                                     num_bits=config.num_bits)

    init_xs, _ = next(g)
    state = init_train_state(weights_key,
                             model=model,
                             init_xs=init_xs,
                             config=config)

    def loss_fn(weights):
        xs, y = next(g)
        xxs = model.apply(weights, xs)[..., 0]
        sq_loss = jnp.mean(jnp.sum((y - xxs)**2, axis=-1), axis=0)
        return sq_loss, (xs, y, xxs)

    @jax.jit
    def train_step(state: train_state.TrainState):
        (loss, aux), grad = jax.value_and_grad(loss_fn,
                                               has_aux=True)(state.params)
        state = state.apply_gradients(grads=grad)
        return loss, aux, state

    writer = tensorboardX.SummaryWriter(logdir=config.logdir)
    for step in range(config.num_train_steps):
        loss, aux, state = train_step(state)
        writer.add_scalar('train/loss', loss, state.step)
        print(f"step {step} loss {loss}")
    writer.flush()


if __name__ == '__main__':
    app.run(main)
