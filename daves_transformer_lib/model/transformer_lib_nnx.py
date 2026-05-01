import math
from typing import Callable, Dict, List, Optional, Tuple, Type

import einx
import jax
import numpy as np
from flax import nnx
from flax.typing import Dtype, Initializer
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class GatedLinearUnit(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation_fn: Callable,
        kernel_init: Initializer,
        param_dtype: Dtype | None,
        rngs: nnx.Rngs,
    ):
        self._activation_fn = activation_fn
        self.linear_proj = nnx.Linear(
            in_features,
            out_features,
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.gate_proj = nnx.Linear(
            in_features,
            out_features,
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.linear_proj(x)
        g = self.gate_proj(x)
        return self._activation_fn(y) * g


class SwiGLUFeedForwardNetwork(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        param_dtype: Dtype | None,
        kernel_init: Initializer,
        rngs: nnx.Rngs,
    ):
        self.up_proj = GatedLinearUnit(
            d_model,
            d_ff,
            activation_fn=nnx.silu,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            d_ff,
            d_model,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.up_proj(x))


class FeedForwardNetwork(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        activation_fn: Callable,
        param_dtype: Dtype | None,
        kernel_init: Initializer,
        rngs: nnx.Rngs,
    ):
        self.activation_fn = activation_fn
        self.up_proj = nnx.Linear(
            d_model,
            d_ff,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            d_ff,
            d_model,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.activation_fn(self.up_proj(x)))


class RotaryPositionalEmbedding(nnx.Module):
    sines: jax.Array
    cosines: jax.Array

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        self._theta = theta
        self._d_k = d_k
        self._max_seq_len = max_seq_len

        # precompute rotation coefficients for each position / frequency combo.
        freqs = theta ** (jnp.arange(0, d_k, 2, dtype=jnp.float32) / d_k)
        pos_enc = jnp.arange(max_seq_len)[..., None] / freqs
        sines = jnp.sin(pos_enc)
        cosines = jnp.cos(pos_enc)

        # duplicate the coefficients since each appears twice in the rotation
        # matrix.
        self.sines = einx.id("n d -> n (d 2)", sines) * (
            (-1.0) ** jnp.arange(1, d_k + 1)
        )
        self.cosines = einx.id("n d -> n (d 2)", cosines)

    def __call__(self, x: jax.Array, token_positions: jax.Array) -> jax.Array:
        # x: [..., seq_len, d_k]
        # token_positions: [..., seq_len]
        x_flipped = einx.flip("... (d [a])", x, a=2)
        return (
            x * self.cosines[token_positions] + x_flipped * self.sines[token_positions]
        )


def scaled_dot_product_attention(
    Q: Float[jax.Array, " ... queries d_k"],
    K: Float[jax.Array, " ... keys d_k"],
    V: Float[jax.Array, " ... keys d_v"],
    mask: Bool[jax.Array, "... queries keys"] | None = None,
) -> Float[jax.Array, "... queries d_v"]:
    sqrt_d = Q.shape[-1] ** 0.5
    scores = einx.dot("... q [d], ... k [d] -> ... q k", Q, K) / sqrt_d
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    return einx.dot("... q [k], ... [k] d -> ... q d", nnx.softmax(scores, axis=-1), V)


class MultiHeadSelfAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None,
        *,
        d_k: int | None = None,
        d_v: int | None = None,
        kernel_init: Initializer,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        self._d_model = d_model
        self._num_heads = num_heads

        d_k = d_model // num_heads if d_k is None else d_k
        d_v = d_k if d_v is None else d_v
        self._d_v = d_v
        self._d_k = d_k

        self.rope = rope

        self.up_proj = nnx.Linear(
            in_features=d_model,
            out_features=num_heads * (d_k + d_k + d_v),
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=num_heads * d_v,
            out_features=d_model,
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Float[Array, "... num_steps d_model"],
        token_positions: Int[Array, "... num_steps"] | None = None,
    ) -> Float[Array, "... num_steps d_model"]:
        qkv = self.up_proj(x)
        Q, K, V = einx.id(
            "... n ((h k1) + (h k2) + (h v)) -> ... h n k1, ... h n k2, ... h n v",
            qkv,
            k1=self._d_k,
            k2=self._d_k,
            v=self._d_v,
        )
        num_steps = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = jnp.arange(num_steps, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mask entries `[i, j]`: token at position i attends to all `j < i`
        # TODO: precompute mask in constructor given max seq len.
        causal_mask = jnp.tril(
            jnp.ones([num_steps, num_steps], dtype=jnp.bool, device=x.device)
        )

        result = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        return self.down_proj(einx.id("... h n v -> ... n (h v)", result))


class Identity(nnx.Module):
    def __call__(self, x):
        return x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        *,
        use_glu: bool = True,
        rmsnorm_position: str = "pre",
        param_dtype=None,
        kernel_init: Initializer,
        rngs: nnx.Rngs,
    ):
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ff = d_ff
        self._rmsnorm_position = rmsnorm_position
        self._use_glu = use_glu

        pre_attn_norm = Identity()
        pre_ffn_norm = Identity()
        post_attn_norm = Identity()
        post_ffn_norm = Identity()
        if rmsnorm_position in ("pre", "both"):
            pre_attn_norm = nnx.RMSNorm(d_model, param_dtype=param_dtype, rngs=rngs)
            pre_ffn_norm = nnx.RMSNorm(d_model, param_dtype=param_dtype, rngs=rngs)
        elif rmsnorm_position in ("post", "both"):
            post_attn_norm = nnx.RMSNorm(d_model, param_dtype=param_dtype, rngs=rngs)
            post_ffn_norm = nnx.RMSNorm(d_model, param_dtype=param_dtype, rngs=rngs)
        elif rmsnorm_position != "none":
            raise ValueError(f"unrecognized rmsnorm_position: {rmsnorm_position}")
        self.pre_attn_norm = pre_attn_norm
        self.pre_ffn_norm = pre_ffn_norm
        self.post_attn_norm = post_attn_norm
        self.post_ffn_norm = post_ffn_norm

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads=num_heads,
            rope=rope,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        if use_glu:
            self.ffn = SwiGLUFeedForwardNetwork(
                d_model=d_model,
                d_ff=d_ff,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                rngs=rngs,
            )
        else:
            self.ffn = FeedForwardNetwork(
                d_model=d_model,
                d_ff=d_ff,
                activation_fn=nnx.silu,
                kernel_init=kernel_init,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(self, x: Float[Array, "... num_steps d_model"]):
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x)))
        return x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))


class TransformerLanguageModel(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        rope_theta=10000.0,
        use_rope=True,
        use_glu=True,
        rmsnorm_position="pre",
        param_dtype=None,
        kernel_init: Initializer = nnx.initializers.glorot_normal(),
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ff = d_ff
        self._use_rope = use_rope
        self._use_glu = use_glu
        self._rmsnorm_position = rmsnorm_position

        self.token_embeddings = nnx.Embed(
            num_embeddings=vocab_size, features=d_model, dtype=param_dtype, rngs=rngs
        )

        rope = None
        if self._use_rope:
            rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length
            )
        self.rope = rope

        self.layers = nnx.List(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=self.rope,
                    use_glu=use_glu,
                    rmsnorm_position=rmsnorm_position,
                    kernel_init=kernel_init,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nnx.RMSNorm(d_model, param_dtype=param_dtype, rngs=rngs)
        self.output_embeddings = nnx.Linear(
            in_features=d_model,
            out_features=vocab_size,
            use_bias=False,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(
        self, tokens: Int[Array, "... num_steps"]
    ) -> Float[Array, "... num_steps vocab_size"]:
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output_embeddings(self.output_norm(x))
