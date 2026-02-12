# transformers


NEXT STEPS:

- have the model and/or generate return a *distribution* over characters
- so then the generation process is an Autoregressive distribution

and/or

- consider training with long-term context
- frame training with the KV cache as an explicitly recurrent process we can differentiate through

SEPARATELY:
- load a real model, like Mistral
- look at RL training with a debate-type objective.




What does it take to load Mistral or LLama? I think that would also force me to figure out the positional embedding stuff.
There's probably not much I can do to improve on either the Mistral codebase (Torch but very clean: https://github.com/mistralai/mistral-src/) or the Levanter (Jax, Equinox, Haliax: https://github.com/stanford-crfm/levanter) codebases for real training work.

But having something that is my own to play with is still nice for weird researchy customization.

