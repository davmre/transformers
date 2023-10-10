import functools
from typing import List, Optional, Tuple

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

import jax
from jax import numpy as jnp

from daves_transformer_lib.transformer_lib import KeyValueCache


@functools.partial(jax.jit, static_argnums=(0,))
def generate_step(
    model,
    key: jax.random.KeyArray,
    weights: PyTree,
    context: Int[Array, "n"],
    context_length: int,
    top_k: Optional[int] = None,
    temperature: Float = 1.0,
    kv_caches: Optional[List[KeyValueCache]] = None,
) -> Tuple[Int[Array, ""], Float[Array, "vocab_size"],
           Optional[List[KeyValueCache]]]:
    forward_key, sample_key = jax.random.split(key)
    all_logits, kv_caches = model.apply(weights,
                                        context,
                                        kv_caches=kv_caches,
                                        rngs=model.rngs(forward_key))
    logits = all_logits[...,  # type: jax.Array
                        context_length - 1, :] / temperature
    if top_k is not None:
        # TODO: find a more efficient top_k implementation.
        v = jnp.sort(logits, axis=-1)[..., -top_k]
        logits = jnp.where(logits < v[..., jnp.newaxis], -jnp.inf,
                           logits)  # type: jax.Array
    probs = jax.nn.softmax(logits)
    c = jax.random.choice(sample_key, logits.shape[-1], shape=(), p=probs)
    return c, jax.nn.log_softmax(logits), kv_caches


def generate(key: jax.random.KeyArray,
             model,
             weights: PyTree,
             context: Array,
             context_length: Optional[int] = None,
             top_k: Optional[int] = None,
             temperature: Float = 1.0,
             include_logprobs: bool = False):

    # Pad context out to block size.
    if context_length is None:
        context_length = len(context)
    assert (context_length > 0)
    context = jnp.concatenate([
        context,
        jnp.zeros([model.block_size - len(context)], dtype=context.dtype)
    ])

    while True:
        gen_key, key = jax.random.split(key)
        c, log_probs, _ = generate_step(model,
                                        key=gen_key,
                                        weights=weights,
                                        context=context,
                                        context_length=context_length,
                                        top_k=top_k,
                                        temperature=temperature,
                                        kv_caches=None)

        # Add the generated character to the context for the next step.
        if context_length < model.block_size:
            context = context.at[context_length].set(c)
        else:
            context = jnp.concatenate(
                [context[1:], jnp.array([c], dtype=context.dtype)])
        context_length = min(context_length + 1, model.block_size)

        if include_logprobs:
            yield c, log_probs
        else:
            yield c


def generate_with_kv_cache(key: jax.random.KeyArray,
                           model,
                           weights: PyTree,
                           context: Array,
                           top_k: Optional[int] = None,
                           temperature: Float = 1.0,
                           include_logprobs: bool = False):

    kv_caches = model.initialize_kv_caches()
    while True:
        gen_key, key = jax.random.split(key)
        c, log_probs, kv_caches = generate_step(model,
                                                key=gen_key,
                                                weights=weights,
                                                context=context,
                                                context_length=len(context),
                                                top_k=top_k,
                                                temperature=temperature,
                                                kv_caches=kv_caches)
        # Pass only the incremental new context since all relevant previous
        # context is stored in the KV cache.
        context = jnp.array([c])

        if include_logprobs:
            yield c, log_probs
        else:
            yield c