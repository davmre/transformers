from abc import ABC
from abc import abstractmethod
from typing import Optional, Tuple

from jaxtyping import Array
from jaxtyping import Int

import jax
from jax import numpy as jnp
import numpy as np


class Dataset(ABC):

    @abstractmethod
    def encode(self, s: str) -> jax.Array:
        pass

    @abstractmethod
    def decode(self, x) -> str:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(
            self, idx: Int
    ) -> Tuple[Int[Array, 'block_size'], Int[Array, 'block_size']]:
        pass


class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from karpathy's minGPT:
    https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
    """

    def __init__(self, data: str, block_size: int):

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.data = data
        self.encoded_data = self.encode(data)

    def encode(self, s: str) -> jax.Array:
        return jnp.array([self.stoi[c] for c in s], dtype=jnp.int32)

    def decode(self, x) -> str:
        return str([self.itos[int(i)] for i in x])

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(
            self, idx: Int
    ) -> Tuple[Int[Array, 'block_size'], Int[Array, 'block_size']]:
        # x = self.encoded_data[idx:idx + self.block_size]
        # y = self.encoded_data[idx + 1:idx + self.block_size + 1]

        x = jax.lax.dynamic_slice(self.encoded_data, [idx], [self.block_size])
        y = jax.lax.dynamic_slice(self.encoded_data, [idx + 1],
                                  [self.block_size])
        return x, y


def character_generator(dataset, batch_size: Optional[int] = None):
    max_idx = len(dataset)

    dataset_get = lambda idx: dataset[idx]
    dataset_get = jax.vmap(dataset_get) if batch_size else dataset_get

    while True:
        idx = np.random.randint(low=0, high=max_idx, size=batch_size)
        x, y = dataset_get(idx)
        yield x, y