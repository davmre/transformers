import dataclasses
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import einx
import jax
import numpy as np
import numpy.typing as npt
import optax
from cs336_basics.optimizers import clip_gradients, get_lr_cosine_schedule
from cs336_basics.utils import Logger
from flax import nnx
from flax.typing import Array, Dtype, Initializer
from jax import numpy as jnp
from jaxtyping import Bool, Float, Int
from simple_parsing import ArgumentParser

from daves_transformer_lib.model.transformer_lib_nnx import TransformerLanguageModel


@dataclass
class TrainState:
    """State tracked by the training loop."""

    step: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    time: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    use_rope: bool = True
    rope_theta: float = 10000.0
    rmsnorm_position: str = "pre"  # "pre", "post", or "none"
    use_glu: bool = True

    def build_model(
        self, dtype: Dtype | None, rngs: nnx.Rngs
    ) -> TransformerLanguageModel:
        return TransformerLanguageModel(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            use_rope=self.use_rope,
            use_glu=self.use_glu,
            rmsnorm_position=self.rmsnorm_position,
            param_dtype=dtype,
            rngs=rngs,
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def save(self, filename: str | Path):
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_string: str) -> "ModelConfig":
        return ModelConfig(**json.loads(json_string))

    @classmethod
    def load(cls, filename: str) -> "ModelConfig":
        with open(filename) as f:
            return cls.from_json(f.read())


@dataclass(frozen=True)
class TrainingConfig:
    token_file: str
    batch_size: int = 32
    max_steps: int | None = None
    learning_rate_max: float = 1e-3
    learning_rate_min: float = 1e-4
    learning_rate_cos_warmup_steps: int = 100
    learning_rate_cos_final_step: int = 5000
    weight_decay: float = 0.01
    print_every: int = 1
    save_every: int = 5
    max_gradient_norm: float | None = None
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    seed: int = 42
    validate_every: int = 5
    validation_set_size: int = 128
    validation_token_file: str | None = None
    device: str | None = None
    dtype: str = "float32"

    def get_dtype(self) -> Dtype:
        return getattr(jnp, self.dtype)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def save(self, filename: str | Path):
        with open(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_string: str) -> "TrainingConfig":
        return TrainingConfig(**json.loads(json_string))

    @classmethod
    def load(cls, filename: str) -> "TrainingConfig":
        with open(filename) as f:
            return cls.from_json(f.read())


@dataclass(frozen=True)
class RunSpec:
    model_config: ModelConfig
    training_config: TrainingConfig
    explicit_run_dir: Path | None = None

    def run_dir(self, base_dir: Path | None = None) -> Path:
        if self.explicit_run_dir:
            return self.explicit_run_dir
        config_hash = hashlib.sha256(
            (repr(self.model_config) + repr(self.training_config)).encode("utf-8")
        )
        run_name = f"run_{str(config_hash.hexdigest())[:8]}"
        if base_dir:
            return base_dir / run_name
        else:
            return Path(run_name)


def loss_fn(
    model: TransformerLanguageModel,
    tokens: Int[Array, "... seq_len"],
    labels: Int[Array, "... seq_len"],
):
    logits = model(tokens)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits


def get_batch(x: npt.NDArray, batch_size: int, context_length: int):
    window_len = context_length + 1
    if window_len + 1 > len(x):
        raise Exception("not enough tokens!")

    # For each batch element, sample a starting point within the sequence.
    max_idx = len(x) - window_len
    batch_idxs = np.random.randint(0, max_idx + 1, size=batch_size)

    # Extract the tokens from each sampled point.
    tokens = jnp.stack([x[idx : idx + window_len] for idx in batch_idxs])
    in_tokens = tokens[:, :-1]
    out_tokens = tokens[:, 1:]
    return in_tokens, out_tokens


def get_batches(token_file: str, batch_size: int, context_length: int):
    tokens = np.memmap(token_file, dtype=np.uint16)
    while True:
        yield get_batch(tokens, batch_size=batch_size, context_length=context_length)


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    tokens: Array,
    labels: Array,
    step: int,
    training_config: TrainingConfig,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, tokens)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(model, grads)


def do_train_step(
    model: nnx.Module,
    optimizer: Optimizer,
    in_tokens: Array,
    out_tokens: Array,
    step: int,
    training_config: TrainingConfig,
):
    model.train()
    optimizer.zero_grad()
    logits = model(in_tokens)
    loss = cross_entropy(logits, out_tokens)
    loss.backward()

    loss_value = loss.item()  # Save loss as Python value for logging.

    max_gradient_norm = training_config.max_gradient_norm
    if max_gradient_norm is None:
        max_gradient_norm = torch.inf
    grad_norm = clip_gradients(model.parameters(), max_l2_norm=max_gradient_norm)

    # Set learning rate for this step.
    learning_rate = get_lr_cosine_schedule(
        step,
        lr_max=training_config.learning_rate_max,
        lr_min=training_config.learning_rate_min,
        warmup_steps=training_config.learning_rate_cos_warmup_steps,
        total_steps=training_config.learning_rate_cos_final_step,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    optimizer.step()
    return loss_value, learning_rate, grad_norm


def run_training(run_spec: RunSpec):
    training_config = run_spec.training_config
    model_config = run_spec.model_config
    run_dir = run_spec.run_dir()
    dtype = training_config.get_dtype()

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging run to {run_dir}")

    logger = Logger(
        config=dataclasses.asdict(training_config) | dataclasses.asdict(model_config),
        log_dir=run_dir,
    )

    training_config.save(run_dir / "training.config")
    model_config.save(run_dir / "model.config")

    rngs = nnx.Rngs(training_config.seed)
    model = model_config.build_model(dtype=dtype, rngs=rngs)

    optimizer = optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=training_config.learning_rate_min,
            peak_value=training_config.learning_rate_max,
            warmup_steps=training_config.learning_rate_cos_warmup_steps,
            decay_steps=training_config.learning_rate_cos_final_step,
            end_value=training_config.learning_rate_min,
        ),
        b1=training_config.adamw_beta1,
        b2=training_config.adamw_beta2,
        weight_decay=training_config.weight_decay,
    )

    if training_config.max_gradient_norm:
        optimizer = optax.chain(
            optax.clip_by_global_norm(training_config.max_gradient_norm), optimizer
        )

    checkpoint_path = run_dir / "model.chk"
    if checkpoint_path.exists():
        train_state = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    else:
        train_state = TrainState()

    validation_set = None
    if training_config.validation_token_file:
        num_validation_batches = (
            training_config.validation_set_size // training_config.batch_size
        )
        validation_loader = get_batches(
            training_config.validation_token_file,
            training_config.batch_size,
            context_length=model_config.context_length,
            device=device,
        )
        validation_set = [
            next(validation_loader) for _ in range(num_validation_batches)
        ]

    initial_time = time.time()
    train_state.time = initial_time
    initial_state = train_state
    last_printed_state = train_state
    for in_tokens, out_tokens in get_batches(
        training_config.token_file,
        batch_size=training_config.batch_size,
        context_length=model_config.context_length,
        device=device,
    ):
        loss_value, learning_rate, grad_norm = do_train_step(
            model=compiled_model,
            optimizer=optimizer,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            step=train_state.step,
            training_config=training_config,
        )

        now = time.time()
        train_state = dataclasses.replace(
            train_state,
            step=train_state.step + 1,
            total_tokens=train_state.total_tokens + out_tokens.numel(),
            time=now,
        )
        logger.log(
            {
                "train/loss": loss_value,
                "learning_rate": learning_rate,
                "tokens": train_state.total_tokens,
                "grad_norm": grad_norm,
            },
            step=train_state.step,
        )
        if train_state.step % training_config.print_every == 0:
            elapsed_sec = train_state.time - last_printed_state.time
            time_per_step = elapsed_sec / training_config.print_every
            tokens_per_step = training_config.batch_size * model_config.context_length
            tokens_per_sec = (
                tokens_per_step * training_config.print_every
            ) / elapsed_sec
            print(
                f"step {train_state.step}: loss {loss_value: .4f} lr {learning_rate: .4f} grad_norm {grad_norm: .1f} time: {train_state.time - initial_state.time: .1f}s ({time_per_step: .1f}s/step) tokens {train_state.total_tokens} toks/s: {tokens_per_sec: .0f}"
            )
            last_printed_state = train_state
        if train_state.step % training_config.validate_every == 0:
            if validation_set:
                batch_losses = [
                    validation_loss(model, in_tokens, out_tokens)
                    for (in_tokens, out_tokens) in validation_set
                ]
                val_loss = np.mean(batch_losses)
                print(f"validation loss: {val_loss}")
                logger.log({"val/loss": val_loss}, step=train_state.step)

        if train_state.step % training_config.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                state=train_state,
                out=checkpoint_path,
            )
            print(f"checkpoint saved to {checkpoint_path}")
        if (
            training_config.max_steps is not None
            and train_state.step >= training_config.max_steps
        ):
            break


def load_config(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("run_spec", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
