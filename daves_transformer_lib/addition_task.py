from enum import Enum
import math
from sys import float_info
from sys import int_info
from typing import Callable, List

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn
from flax import struct

from daves_transformer_lib import transformer_lib


@struct.dataclass
class ArithmeticProblem:
    a: jax.Array
    b: jax.Array
    op: Callable

    @property
    def result(self):
        return self.op(self.a, self.b)


def random_addition_problems(key,
                             batch_size,
                             max_input=2**8,
                             min_input=0,
                             dtype=jnp.int32):
    if max_input > np.iinfo(dtype).max:
        raise ValueError(
            f'Max input {max_input} is too large for dtype {dtype}')

    inputs = jax.random.randint(key,
                                shape=[2, batch_size],
                                minval=min_input,
                                maxval=max_input,
                                dtype=dtype)
    return ArithmeticProblem(a=inputs[0], b=inputs[1], op=lambda a, b: a + b)


def data_generator(key, batch_size, num_bits):
    encode = jax.vmap(lambda n: int_to_binary_encoding(n, num_bits=num_bits))
    while True:
        key, this_key = jax.random.split(key, 2)
        problems = random_addition_problems(this_key,
                                            batch_size=batch_size,
                                            max_input=2**(num_bits - 1))

        x1 = encode(problems.a)
        x2 = encode(problems.b)
        y = encode(problems.result)
        yield jnp.stack([x1, x2], axis=-1), y


def int_to_binary_encoding(n, num_bits):
    powers_of_two = 2**jnp.arange(num_bits)[::-1]
    return (n // powers_of_two) % 2


def binary_encoding_to_int(x):
    num_bits = x.shape[-1]
    powers_of_two = 2**jnp.arange(num_bits)[::-1]
    return sum(x * powers_of_two)


class AdditionModel(nn.Module):
    num_heads: int
    num_iters: int
    d_head: int
    d_ff: int
    dropout: nn.Module
    internal_dtype = jnp.float32

    @nn.compact
    def __call__(self, xs):
        d_model = self.d_head * self.num_heads
        num_bits = xs.shape[-2]

        positions = self.param('position_embeddings', jax.random.normal,
                               (num_bits, d_model))
        zs = nn.Dense(d_model, dtype=self.internal_dtype)(xs) + positions

        transformer_layer = transformer_lib.EncoderLayer(
            size=d_model,
            self_attn=transformer_lib.MultiHeadedAttention(
                self.num_heads, d_model, dropout=self.dropout),
            feed_forward=transformer_lib.PositionwiseFeedForward(
                d_model, self.d_ff, dropout=self.dropout),
            dropout=self.dropout)
        for _ in range(self.num_iters):
            zs = transformer_layer(zs, mask=None)

        ys = nn.Dense(1, dtype=self.internal_dtype)(zs)
        return ys
