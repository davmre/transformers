from collections import defaultdict
import functools

from ml_collections import config_dict
import tensorboardX
import tree

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import optax


def save_checkpoint_callback(checkpoint_dir):

    def fn(_xs, _y, _loss, _grad, _aux, state):
        _ = checkpoints.save_checkpoint(checkpoint_dir,
                                        target=state,
                                        step=state.step)
        print("saved checkpoint")

    return fn


class Trainer:

    @staticmethod
    def get_default_config():
        config = config_dict.ConfigDict()
        config.num_train_steps = 6
        config.resume_from_checkpoint = -1
        config.checkpoint_dir = None
        config.log_dir = '/tmp/logs'
        return config

    def __init__(self,
                 config,
                 model,
                 data_generator,
                 loss_fn,
                 checkpoint_dir=None,
                 checkpoint_interval=0):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.data_generator = data_generator
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self._callbacks = []
        if checkpoint_dir:
            self.add_callback(step_interval=checkpoint_interval,
                              fn=save_checkpoint_callback(checkpoint_dir))

    def add_callback(self, step_interval, fn):
        self._callbacks.append((step_interval, fn))

    def init_parameters(self, key):
        model = self.model
        init_xs, _ = next(self.data_generator)
        params_key, rngs_key = jax.random.split(key)
        rngs = model.rngs(rngs_key)
        rngs = dict(rngs, params=params_key)
        return model.init(rngs, init_xs)

    def init(self, parameters, optimizer):
        return train_state.TrainState(step=0,
                                      apply_fn=self.model.apply,
                                      params=parameters,
                                      tx=optimizer,
                                      opt_state=optimizer.init(parameters))

    def init_from_checkpoint(self, step, optimizer):
        parameters = self.init_parameters(jax.random.PRNGKey(0))
        state = self.init(parameters, optimizer)
        return checkpoints.restore_checkpoint(self.checkpoint_dir,
                                              target=state,
                                              step=step)

    def forward_loss(self, parameters, key, xs, y):
        y_pred = self.model.apply(parameters, xs, rngs=self.model.rngs(key))
        loss = self.loss_fn(y, y_pred)
        # Average over any batch and position dimensions.
        scalar_loss = jnp.mean(loss)
        return scalar_loss, (xs, y, y_pred, loss)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: train_state.TrainState, key, xs, y):
        key, next_key = jax.random.split(key)
        (loss,
         aux), grad = jax.value_and_grad(functools.partial(self.forward_loss,
                                                           key=key,
                                                           xs=xs,
                                                           y=y),
                                         has_aux=True)(state.params)
        state = state.apply_gradients(grads=grad)
        return loss, grad, aux, state, next_key

    def run(self, key, state):
        writer = tensorboardX.SummaryWriter(logdir=self.config.log_dir)

        for _ in range(self.config.num_train_steps):
            xs, y = next(self.data_generator)
            loss, grad, aux, state, key = self.step(key=key,
                                                    state=state,
                                                    xs=xs,
                                                    y=y)

            for i, fn in self._callbacks:
                if state.step % i == 0:
                    fn(xs, y, loss, grad, aux, state)
            writer.add_scalar('train/loss', loss, state.step)
        writer.flush()

        return state