# Repository Guidelines

This is a research codebase for transformer language models in JAX. It aims to
be simple, understandable, and easily modifiable for testing new ideas.

## Build, Test, and Development Commands
Use a virtual environment:
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install -r requirements-dev.txt`

Key workflows:
- `python3 experiments/character_language_model.py` — run the character-level training experiment locally.
- `python3 -m pytest daves_transformer_lib -q` — run unit tests.
- `ruff check .` — run lint/import checks.
- `pyright` — run static type checking.
- `modal run experiments/character_language_model.py` — run remote Modal job (if configured).
