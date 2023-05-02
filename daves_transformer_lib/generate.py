import functools

import jax
from jax import numpy as jnp


@functools.partial(jax.jit, static_argnums=(0,))
def generate_step(model,
                  key,
                  weights,
                  context,
                  context_length,
                  top_k=None,
                  temperature=1.0):
    forward_key, sample_key = jax.random.split(key)
    all_logits = model.apply(weights, context, rngs=model.rngs(forward_key))
    logits = all_logits[..., context_length - 1, :] / temperature
    if top_k is not None:
        # TODO: find a more efficient top_k implementation.
        v = jnp.sort(logits, axis=-1)[..., -top_k]
        logits = jnp.where(logits < v[..., jnp.newaxis], -jnp.inf, logits)
    probs = jax.nn.softmax(logits)
    c = jax.random.choice(sample_key, logits.shape[-1], shape=(), p=probs)
    return c, jax.nn.log_softmax(logits)


def generate(key,
             model,
             weights,
             context,
             num_tokens,
             context_length=None,
             top_k=None,
             temperature=1.0,
             include_logprobs=False):

    # Pad context out to block size.
    if context_length is None:
        context_length = len(context)
    assert (context_length > 0)
    context = jnp.concatenate([
        context,
        jnp.zeros([model.block_size - len(context)], dtype=context.dtype)
    ])

    for t in range(num_tokens):
        gen_key, key = jax.random.split(key)
        c, log_probs = generate_step(model,
                                     key=gen_key,
                                     weights=weights,
                                     context=context,
                                     context_length=context_length,
                                     top_k=top_k,
                                     temperature=temperature)

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

    return context