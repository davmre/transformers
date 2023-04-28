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
# from flax.training import checkpoints
from flax.training import train_state
import optax

from daves_transformer_lib import addition_task
from daves_transformer_lib import train

# Example invocation:
# `python3 experiments/addition.py --ml_config.num_train_steps=4``
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
config.resume_from_checkpoint = -1
config.seed = 0
config.checkpoint_dir = '/tmp/addition_checkpoints'
config.logdir = '/tmp/addition_logs'
_CONFIG = config_flags.DEFINE_config_dict('ml_config', config)


def init_train_state(key, model, init_xs, opt):
    weights = model.init(key, init_xs[0, ...])
    return train_state.TrainState(step=0,
                                  apply_fn=model.apply,
                                  params=weights,
                                  tx=opt,
                                  opt_state=opt.init(weights))


def squared_loss(y, y_pred):
    y_pred = y_pred[..., 0]  # TODO: put this somewhere more principled?
    return jnp.mean(jnp.sum((y - y_pred)**2, axis=-1), axis=0)


def main(_):
    config = _CONFIG.value
    key = jax.random.PRNGKey(0)

    data_key, train_key = jax.random.split(key)

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

    trainer = train.Trainer(config=config,
                            model=model,
                            optimizer=optax.adam(3e-3),
                            data_generator=g,
                            loss_fn=squared_loss)
    trainer.run(train_key)


if __name__ == '__main__':
    import sys
    print("ARGV", sys.argv)
    app.run(main)
