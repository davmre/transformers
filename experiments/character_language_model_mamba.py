import functools
import itertools

from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
import modal
import tree

import jax
from jax import config as jax_config
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax
import orbax.checkpoint

from daves_transformer_lib import data_generators
from daves_transformer_lib import generate
from daves_transformer_lib import mamba_lib
from daves_transformer_lib import train

config = config_dict.ConfigDict()
config.model = config_dict.ConfigDict()
config.model.num_layers = 6
config.model.d_model = 96
config.train = config_dict.ConfigDict()
config.train.weight_decay = 0.1
config.train.num_steps = 5000
config.train.checkpoint_interval = 500
config.train.log_dir = '/tmp/mamba'
config.batch_size = 64
config.seed = 0

_CONFIG = config_flags.DEFINE_config_dict('ml_config', config)

jax_config.update("jax_debug_nans", True)

# Remote training with Modal: follow getting-started instructions at
# https://modal.com/home
# to install the client and create a token. Then run
# `modal run experiments/character_language_model.py`
# To download trained model checkpoints, run
# ```
# modal volume get chargpt-checkpoints ** remote
# modal volume get chargpt-checkpoints "**/.*" remote
# ```
# (TODO: is there a single glob pattern to recursively download hidden and
# non-hidden files? )
stub = modal.Stub(name="charmamba")
image = modal.Image.debian_slim().pip_install_from_requirements(
    '/Users/dave/code/transformers/requirements.txt')
volume = modal.NetworkFileSystem.new().persisted("charmamba-checkpoints")


def weight_decay_mask(params):

    def f(p, x):
        # Apply weight decay only to weight matrices of Dense layers.
        weight_decay = (p[-1] == 'kernel')
        return weight_decay

    return tree.map_structure_with_path(f, params)


@stub.function(image=image,
               network_file_systems={"/tmp/mamba": volume},
               gpu="any",
               timeout=3600)
def do_training(config, text):
    key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)  # For data loading.

    print("Default backend", jax.default_backend())
    print("Devices", jax.devices())

    # construct the training dataset
    train_dataset = data_generators.CharDataset(text, block_size=128)
    g = data_generators.character_generator(train_dataset,
                                            batch_size=config.batch_size)

    model = mamba_lib.MambaLanguageModel(n_tokens=train_dataset.vocab_size,
                                         n_layers=config.model.num_layers,
                                         d_model=config.model.d_model,
                                         expand=2,
                                         d_conv=4,
                                         d_state=8)

    trainer = train.Trainer(
        config=config.train,
        model=model,
        data_generator=g,
        loss_fn=jax.vmap(train.log_loss, in_axes=(0, 0)),
        log_dir=config.train.log_dir,
        checkpoint_options=orbax.checkpoint.CheckpointManagerOptions(
            save_interval_steps=config.train.checkpoint_interval,
            create=True,
            max_to_keep=3))
    key, init_key = jax.random.split(key)
    params = trainer.init_parameters(key=init_key)

    n_params = sum([np.prod(p.shape) for p in tree.flatten(params)])
    print("number of parameters: %.2fM" % (n_params / 1e6,))

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(5e-4,
                    b1=0.9,
                    b2=0.95,
                    weight_decay=config.train.weight_decay,
                    mask=weight_decay_mask(params)),
    )
    state = trainer.init(params, optimizer)

    def print_loss(_xs, _y, loss, _aux, state):
        print(f"step {state.step} loss {loss}")

    trainer.add_callback(step_interval=1, fn=print_loss)

    state = trainer.run(key=key, state=state, num_steps=config.train.num_steps)


@stub.local_entrypoint()
def modal_main():
    #config = _CONFIG.value
    global config
    with open('experiments/data/shakespear.txt', 'r') as f:
        text = f.read()
    do_training.remote(config, text)


def main(_):
    with open('experiments/data/shakespear.txt', 'r') as f:
        text = f.read()
    config = _CONFIG.value
    do_training.local(config, text)


if __name__ == '__main__':
    app.run(main)
