from typing import Callable, Dict, List, Optional, Tuple

from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import PyTree

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn


def parallel_scan_op_fn(a, b):
    # Parallel scan for SSMs described in
    # Smith, Warrington, Linderman (2023, https://arxiv.org/abs/2208.04933)
    # Appendix H
    Ai, Bxi = a
    Aj, Bxj = b
    return (Aj * Ai, Aj * Bxi + Bxj)


def inverse_softplus(x):
    return x + jnp.log(-jnp.expm1(-x))


def s4d_real_initializer(key, shape):
    del key
    N = shape[-1]
    return jnp.tile(jnp.log((jnp.arange(N) + 1.)), tuple(shape[:-1]) + (1,))


def make_delta_initializer(dt_min=0.001, dt_max=1.0):

    def delta_initializer(key, shape):
        dt = jnp.exp(
            jax.random.uniform(key, shape) *
            (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min))
        return inverse_softplus(dt)

    return delta_initializer


class SelectiveSSM(nn.Module):
    N: int  # num_state_dims_per_channel
    D: int  # num_channels

    def setup(self):
        self.s_B = nn.Dense(self.N)
        self.s_C = nn.Dense(self.N)
        self.s_Delta1 = nn.Dense(1)
        self.log_neg_Adiag = self.param("log_neg_Adiag", s4d_real_initializer,
                                        [self.D, self.N])
        self.delta_param = self.param("delta_param", make_delta_initializer(),
                                      [self.D])

    def _local_params(self, xs: Float[Array, "num_positions num_channels"]):
        Bs = self.s_B(xs[..., jnp.newaxis])  # [L, D, N]
        Cs = self.s_C(xs[..., jnp.newaxis])  # [L, D, N]
        Delta = jax.nn.softplus(self.delta_param + self.s_Delta1(xs))  # [L, D]
        return Bs, Cs, Delta

    def _discretize(self, Delta, B):
        # N: hidden dim
        # D: number of input channels
        #
        # Args:
        #  Delta: [..., D]
        #  B: [..., N]
        # Returns:
        #  Abar_diag: [..., D, N]
        #  Bbar: [..., D, N]
        Adiag = -jnp.exp(self.log_neg_Adiag)
        log_Abar_diag = Delta[..., jnp.newaxis] * Adiag  # [D, N]
        Abar_diag = jnp.exp(log_Abar_diag)
        Bbar = 1. / log_Abar_diag * (Abar_diag - 1.) * Delta[...,
                                                             jnp.newaxis] * B
        return Abar_diag, Bbar

    def __call__(
        self,
        xs: Float[Array, "num_positions num_channels"],
        initial_hidden: Optional[Float[
            Array, "num_channels num_state_dims_per_channel"]] = None
    ) -> Tuple[Float[Array, "num_positions num_channels"], Float[
            Array, "num_positions num_channels num_state_dims_per_channel"]]:
        Bs, Cs, Delta = self._local_params(xs)
        Abar_diag, Bbars = self._discretize(Delta, Bs)

        Bxs = Bbars * xs[..., jnp.newaxis]  # [L, D, N]
        if initial_hidden is not None:
            Bxs = Bxs.at[0, ...].add(Abar_diag[0] * initial_hidden)

        _, hs = jax.lax.associative_scan(parallel_scan_op_fn,
                                         (Abar_diag, Bxs))  # [L, D, N]
        ys = jnp.sum(hs * Cs, axis=-1)
        return ys, hs

    def step(self, h_t_minus_1, x_t):
        # x_t: [D]
        # h_t: [D, N]
        B, C, Delta = self._local_params(x_t)
        Abar_diag, Bbar = self._discretize(Delta, B)

        h_t = Abar_diag * h_t_minus_1 + Bbar * x_t[..., jnp.newaxis]
        y_t = jnp.sum(C * h_t, axis=-1)
        return h_t, y_t


class MambaBlock(nn.Module):
    d_model: int
    d_conv: int
    expand: int
    ssm_layer: SelectiveSSM

    def setup(self):
        d_inner = self.d_model * self.expand
        self.input_proj = nn.Dense(2 * d_inner)
        self.output_proj = nn.Dense(self.d_model)
        self.conv_layer = nn.Conv(features=d_inner,
                                  kernel_size=[self.d_conv],
                                  padding='CAUSAL')

    def __call__(self, xs, initial_hidden=None):
        # xs: [L, D]
        # initial_hidden: [D, ssm_layer.N]
        activation_fn = jax.nn.silu

        xzs = self.input_proj(xs)  # [L, 2DE]
        xs, zs = jnp.split(xzs, 2, axis=-1)  # each [L, DE]

        xs = activation_fn(self.conv_layer(xs))
        xs, _ = self.ssm_layer(xs, initial_hidden=initial_hidden)

        return self.output_proj(xs * activation_fn(zs))


class ResidualBlock(nn.Module):
    layer: nn.Module

    def setup(self):
        self.norm = nn.LayerNorm()

    def __call__(self, xs):
        # xs: [L, D]
        ys = self.layer(xs)
        return ys + self.norm(ys)


class MambaLanguageModel(nn.Module):
    n_tokens: int
    n_layers: int
    d_model: int
    expand: int
    d_state: int
    d_conv: int
    internal_dtype = jnp.float32

    def rngs(self, key: jax.random.KeyArray):
        return {}

    @nn.compact
    def __call__(self, xs):
        # xs: [L] ints
        num_positions = xs.shape[-1]
        token_embed = nn.Embed(num_embeddings=self.n_tokens,
                               features=self.d_model,
                               embedding_init=nn.initializers.normal(0.02))
        zs = token_embed(xs)

        for layer_idx in range(self.n_layers):
            layer = ResidualBlock(
                MambaBlock(d_model=self.d_model,
                           d_conv=self.d_conv,
                           expand=self.expand,
                           ssm_layer=SelectiveSSM(D=self.d_model * self.expand,
                                                  N=self.d_state)))
            zs = layer(zs)

        zs = nn.LayerNorm()(zs)
        ys = nn.Dense(self.n_tokens,
                      use_bias=False,
                      dtype=self.internal_dtype,
                      kernel_init=nn.initializers.normal(0.02))(zs)
        return ys, None
