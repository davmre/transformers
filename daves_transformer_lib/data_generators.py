from jax import numpy as jnp
import numpy as np


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
