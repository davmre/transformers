# transformers


Cleanup and modernization:

- separate model/training from tasks and data generation
- einx for nice shape handling
- swiglu / rope
- flashattention
- tokenizer, and do all work on tokenized sequences
- nnx rather than linen
- and then also to get running with a real model




### Old

NEXT STEPS:

- have the model and/or generate return a *distribution* over characters
- so then the generation process is an Autoregressive distribution

and/or

- consider training with long-term context
- frame training with the KV cache as an explicitly recurrent process we can differentiate through

SEPARATELY:
- load a real model, like Mistral
- look at RL training with a debate-type objective.