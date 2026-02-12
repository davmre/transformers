from typing import Optional, Tuple

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float


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
    return jnp.tile(jnp.log(jnp.arange(1, N + 1)), tuple(shape[:-1]) + (1,))


class SelectiveSSM(nn.Module):
    d_state: int  # N
    d_inner: int  # D
    dt_rank: int = 1
    dt_scale: float = 1.0
    dt_min: float = 0.001
    dt_max: float = 1.0

    def _dt_weight_initializer(self, key, shape, dtype=jnp.float32):
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        return jax.random.uniform(key,
                                  shape,
                                  dtype=dtype,
                                  minval=-dt_init_std,
                                  maxval=dt_init_std)

    def _dt_bias_initializer(self, key, shape, dtype=jnp.float32):
        dt = jnp.exp(
            jax.random.uniform(key, shape, dtype=dtype) *
            (jnp.log(self.dt_max) - jnp.log(self.dt_min)) +
            jnp.log(self.dt_min))
        return inverse_softplus(dt)

    def setup(self):
        self.x_proj = nn.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(self.d_inner,
                                kernel_init=self._dt_weight_initializer,
                                bias_init=self._dt_bias_initializer,
                                use_bias=True)
        self.log_neg_Adiag = self.param("log_neg_Adiag", s4d_real_initializer,
                                        [self.d_inner, self.d_state])

    def _local_params(self, xs: Float[Array, "num_positions num_channels"]):
        proj = self.x_proj(xs)  # [L, 2N + 1]
        Bs, Cs, delta_tmp = jnp.split(proj,
                                      np.cumsum([self.d_state, self.d_state]),
                                      axis=-1)
        Delta = jax.nn.softplus(self.dt_proj(delta_tmp))  # [L, D]
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
        log_Abar_diag = Delta[..., jnp.newaxis] * Adiag  # [..., D, N]
        Abar_diag = jnp.exp(log_Abar_diag)
        Bbar = (1. / log_Abar_diag * (Abar_diag - 1.) *
                Delta[..., jnp.newaxis] * B[..., jnp.newaxis, :])
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
        ys = jnp.sum(hs * Cs[..., jnp.newaxis, :], axis=-1)
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
                                  feature_group_count=d_inner,
                                  padding='CAUSAL')
        self.passthrough_param = self.param('passthrough',
                                            nn.initializers.constant(1.), ())

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


class MambaLanguageModel(nn.Module):
    n_tokens: int
    n_layers: int
    d_model: int
    expand: int
    d_state: int
    d_conv: int
    internal_dtype = jnp.float32

    def rngs(self, key: jax.Array):
        return {}

    @nn.compact
    def __call__(self, xs, hidden_context=None):
        # xs: [L] ints
        token_embed = nn.Embed(num_embeddings=self.n_tokens,
                               features=self.d_model,
                               embedding_init=nn.initializers.normal(0.02))
        zs = token_embed(xs)

        for layer_idx in range(self.n_layers):
            layer = ResidualBlock(
                MambaBlock(d_model=self.d_model,
                           d_conv=self.d_conv,
                           expand=self.expand,
                           ssm_layer=SelectiveSSM(d_inner=self.d_model *
                                                  self.expand,
                                                  d_state=self.d_state)))
            zs = layer(zs)

        zs = nn.LayerNorm()(zs)
        ys = nn.Dense(self.n_tokens,
                      use_bias=False,
                      dtype=self.internal_dtype,
                      kernel_init=nn.initializers.normal(0.02))(zs)
        return ys, None
