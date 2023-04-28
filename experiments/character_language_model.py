import functools

from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
import orbax.checkpoint
import tree

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
import optax

from daves_transformer_lib import generate
from daves_transformer_lib import gpt_model
from daves_transformer_lib import train

config = config_dict.ConfigDict()
config.model = gpt_model.GPTModel.get_default_config()
config.train = config_dict.ConfigDict()
config.train.weight_decay = 0.1
config.train.num_steps = 5000
config.train.checkpoint_interval = 50
config.train.log_dir = '/tmp/gpt'
config.batch_size = 64
config.seed = 0

_CONFIG = config_flags.DEFINE_config_dict('ml_config', config)


class CharDataset():
    """
    Emits batches of characters.

    Adapted from karpathy's minGPT:
    https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
    """

    def __init__(self, data, block_size):

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.data = data

    def encode(self, s):
        return jnp.array([self.stoi[c] for c in s], dtype=jnp.int32)

    def decode(self, x):
        return str([self.itos[int(i)] for i in x])

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = jnp.array(dix[:-1], dtype=jnp.int32)
        y = jnp.array(dix[1:], dtype=jnp.int32)
        return x, y


def character_generator(dataset):
    max_idx = len(dataset)
    while True:
        idx = np.random.randint(low=0, high=max_idx)
        x, y = dataset[idx]
        yield x, y


def batch_generator(g, batch_size):
    while True:
        xs, ys = zip(*[next(g) for _ in range(batch_size)])
        yield (jnp.stack(xs, axis=0), jnp.stack(ys, axis=0))


def main(_):
    config = _CONFIG.value
    key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)  # For data loading.

    # construct the training dataset
    with open('experiments/data/shakespear.txt', 'r') as f:
        text = f.read()

    train_dataset = CharDataset(text, block_size=128)
    g = character_generator(train_dataset)
    g = batch_generator(g, batch_size=config.batch_size)

    model = gpt_model.GPTModel(vocab_size=train_dataset.vocab_size,
                               num_heads=config.model.num_heads,
                               num_layers=config.model.num_layers,
                               d_head=config.model.d_head,
                               d_ff=config.model.d_ff,
                               block_size=train_dataset.block_size,
                               dropout=nn.Dropout(
                                   rate=config.model.dropout_rate,
                                   deterministic=False))

    trainer = train.Trainer(
        config=config.train,
        model=model,
        data_generator=g,
        loss_fn=jax.vmap(gpt_model.log_loss, in_axes=(0, 0)),
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
                    mask=gpt_model.weight_decay_mask(params)),
    )
    state = trainer.init(params, optimizer)

    def generate_text(_xs,
                      _y,
                      _loss,
                      _grad,
                      _aux,
                      state,
                      context=' ',
                      verbose=False):
        tokens_with_lps = generate.generate(
            jax.random.PRNGKey(0),
            model=model,
            weights=state.params,
            context=train_dataset.encode(context),
            top_k=10,
            num_tokens=500)

        if verbose:
            print(f"step {state.step}: ")
            for t, lps in tokens_with_lps:
                for c, i in train_dataset.stoi.items():
                    print(f" char `{repr(c)}` (i={i}): {lps[i]:.2f}", end='')
                print()
                print("actually sampled: ", repr(train_dataset.itos[int(t)]),
                      "lp", lps[t])
            print()
        else:
            tokens = [int(t) for t, lps in tokens_with_lps]
            print(f"step {state.step}: ")
            for t in tokens:
                print(train_dataset.itos[t], end='')
            print()

    def print_loss(_xs, _y, loss, _grad, _aux, state):
        print(f"step {state.step} loss {loss}")

    trainer.add_callback(step_interval=25,
                         fn=functools.partial(generate_text,
                                              context="O God, O God!"))
    trainer.add_callback(step_interval=1, fn=print_loss)

    state = trainer.run(key=key, state=state, num_steps=config.train.num_steps)


if __name__ == '__main__':
    app.run(main)
