from enum import Enum
import math
from typing import Callable, List

import jax
from jax import numpy as jnp

from flax import linen as nn
from flax import struct


@struct.dataclass
class ArithmeticProblem:
    a: jnp.ndarray
    b: jnp.ndarray
    op: Callable

    @property
    def result(self):
        return self.op(self.a, self.b)


def random_addition_problems(key, batch_size, max_input=2**8, min_input=0):
    inputs = jax.random.randint(key,
                                shape=[2, batch_size],
                                minval=min_input,
                                maxval=max_input)
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
