from typing import Callable, Optional, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float

###############################################################################
# Work-in-progress implementation of the BASED recipe for linear attention:
# https://hazyresearch.stanford.edu/blog/2024-03-03-based
# https://arxiv.org/abs/2402.18668
###############################################################################


def softmax_taylor_exp_feature_map(x: Float[Array, "... d"]):
    d = x.shape[-1]
    rd = jnp.sqrt(d)
    rrd = jnp.sqrt(rd)

    x2 = jnp.reshape(
        x[..., jnp.newaxis, :] * x[..., jnp.newaxis], x.shape[:-1] + (1,)
    ) / jnp.sqrt(2.0)
    return jnp.concatenate([jnp.ones(x[..., :1].shape), x / rrd, x2 / rd], axis=-1)


class LinearAttention(nn.Module):
    d_model: int
    key_dim: int
    value_dim: int
    num_heads: int
    feature_map: Callable
    num_queries_per_head: int = 1  # Allows multi-query attention.
    eps: float = 1e-10

    def setup(self):
        self.proj_q = nn.Dense(
            self.num_heads * self.num_queries_per_head * self.key_dim
        )
        self.proj_k = nn.Dense(self.num_heads * self.key_dim)
        self.proj_v = nn.Dense(self.num_heads * self.value_dim)
        self.proj_o = nn.Dense(self.d_model)

    def __call__(
        self,
        xs: Float[Array, "num_positions d_model"],
        initial_hidden: Optional[Float[Array, "num_heads key_dim value_dim"]] = None,
    ) -> Tuple[
        Float[Array, "num_positions d_model"],
        Float[Array, "num_positions num_channels num_state_dims_per_channel"],
    ]:
        # Project input into keys, queries, values.
        q, k, v = self.proj_q(xs), self.proj_k(xs), self.proj_v(xs)
        q = jnp.reshape(
            q, [-1, self.num_queries_per_head, self.num_heads, self.key_dim]
        )
        k = jnp.reshape(k, [-1, self.num_heads, self.key_dim])
        v = jnp.reshape(v, [-1, self.num_heads, self.value_dim])

        q, k = self.feature_map(q), self.feature_map(k)

        # Accumulate hidden state matrix across positions.
        q = q[..., jnp.newaxis, :]  # [N, Q, H, 1, K]
        k = k[..., jnp.newaxis, :, jnp.newaxis, :]  # [N, 1, H, 1, K]
        v = v[..., jnp.newaxis, :, :, jnp.newaxis]  # [N, 1, H, V, 1]
        hiddens = jnp.cumsum(k * v, axis=0) / (
            jnp.cumsum(k, axis=0) + self.eps
        )  # [N, 1, H, V, K]
        if initial_hidden is not None:
            hiddens += initial_hidden[jnp.newaxis, :, :, :]

        # Apply queries.
        results = jnp.sum(q * hiddens, axis=-1)  # [N, Q, H, V]
        # Project back to the original model dimension.
        ys = self.proj_o(
            jnp.reshape(
                results,
                [-1, self.num_queries_per_head * self.num_heads * self.value_dim],
            )
        )

        return ys, hiddens[:, 0, :, :, :]


class BasedBlock(nn.Module):
    d_model: int
    d_conv: int
    expand: int
    attention_layer: LinearAttention

    def setup(self):
        d_inner = self.d_model * self.expand
        self.input_proj = nn.Dense(2 * d_inner)
        self.output_proj = nn.Dense(self.d_model)
        self.conv_layer = nn.Conv(
            features=d_inner,
            kernel_size=[self.d_conv],
            feature_group_count=d_inner,
            padding="CAUSAL",
        )
        self.passthrough_param = self.param(
            "passthrough", nn.initializers.constant(1.0), ()
        )

    def __call__(self, xs, initial_hidden=None):
        # xs: [L, D]
        # initial_hidden: [D, ssm_layer.N]
        activation_fn = jax.nn.silu

        xzs = self.input_proj(xs)  # [L, 2DE]
        xs, zs = jnp.split(xzs, 2, axis=-1)  # each [L, DE]

        xs = activation_fn(self.conv_layer(xs))
        ys, _ = self.ssm_layer(xs, initial_hidden=initial_hidden)
        ys = ys + self.passthrough_param * xs
        return self.output_proj(ys * activation_fn(zs))


class ResidualBlock(nn.Module):
    layer: nn.Module

    def setup(self):
        self.norm = nn.LayerNorm()

    def __call__(self, xs):
        # xs: [L, D]
        ys = self.layer(xs)
        return self.norm(xs + ys)
