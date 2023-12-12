from collections import defaultdict
import functools
import os
import time
import typing

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree
import tensorboardX

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint


@functools.partial(jnp.vectorize, signature='(),(k)->()')
def log_loss(y: Int[Array, "..."],
             y_pred: Float[Array, "... k"]) -> Float[Array, "..."]:
    # Allow a slight type mismatch: y_pred are unnormalized logits,
    # while `y` is an integer index.
    assert (y.shape == ())
    logits = jax.nn.log_softmax(y_pred)
    return -logits[y]


class Trainer:

    def __init__(self,
                 config,
                 model,
                 data_generator,
                 loss_fn,
                 checkpoint_options=None,
                 log_dir="/tmp/training_log"):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.data_generator = data_generator
        self.log_dir = log_dir
        self._callbacks = []

        checkpoint_manager = None
        if checkpoint_options:
            checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                checkpoint_dir,
                orbax.checkpoint.Checkpointer(
                    orbax.checkpoint.PyTreeCheckpointHandler()),
                checkpoint_options)
        self.checkpoint_manager = checkpoint_manager

    def add_callback(self, step_interval, fn):
        self._callbacks.append((step_interval, fn))

    def init_parameters(self, key):
        model = self.model
        init_xs, _ = next(self.data_generator)
        params_key, rngs_key = jax.random.split(key)
        rngs = model.rngs(rngs_key)
        rngs = dict(rngs, params=params_key)
        return model.init(rngs, init_xs[0, ...])

    def init(self, parameters: PyTree,
             optimizer: optax.GradientTransformation) -> train_state.TrainState:
        state = train_state.TrainState(step=0,
                                       apply_fn=self.model.apply,
                                       params=parameters,
                                       tx=optimizer,
                                       opt_state=optimizer.init(parameters))
        if self.checkpoint_manager is not None:
            resume_from_step = self.checkpoint_manager.latest_step()
            if resume_from_step is not None:
                state = self.checkpoint_manager.restore(resume_from_step,
                                                        items=state)
        return typing.cast(train_state.TrainState, state)

    def forward_loss(self, parameters: PyTree, key: jax.random.KeyArray,
                     xs: PyTree, y: PyTree):
        # Apply the model in parallel to each batch element.
        batch_size = xs.shape[0]
        split_rngs = jax.tree_map(lambda k: jax.random.split(k, batch_size),
                                  self.model.rngs(key))
        y_pred, _ = jax.vmap(lambda w, x, r: self.model.apply(w, x, rngs=r),
                             in_axes=(None, 0, 0))(parameters, xs, split_rngs)
        loss = self.loss_fn(y, y_pred)
        # Average over any batch and position dimensions.
        scalar_loss = jnp.mean(loss)
        return scalar_loss, (xs, y, y_pred, loss)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: train_state.TrainState, key: jax.random.KeyArray,
             xs: PyTree, y: PyTree):
        key, next_key = jax.random.split(key)
        (loss,
         aux), grad = jax.value_and_grad(functools.partial(self.forward_loss,
                                                           key=key,
                                                           xs=xs,
                                                           y=y),
                                         has_aux=True)(state.params)
        state = state.apply_gradients(grads=grad)
        return loss, aux, state, next_key

    def run(self, key: jax.random.KeyArray, state: train_state.TrainState,
            num_steps: int) -> train_state.TrainState:
        writer = tensorboardX.SummaryWriter(logdir=self.log_dir)

        for _ in range(num_steps):
            t0 = time.time()
            xs, y = next(self.data_generator)
            t01 = time.time()
            print(f"data {t01 - t0:.3f}")
            loss, aux, state, key = self.step(key=key, state=state, xs=xs, y=y)
            t1 = time.time()
            print(f"{t1 - t0:.3f}")

            for i, fn in self._callbacks:
                if state.step % i == 0:
                    fn(xs, y, loss, aux, state)
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(int(state.step),
                                             state,
                                             metrics={'loss': loss})
            writer.add_scalar('train/loss', loss, int(state.step))
        writer.flush()

        return state