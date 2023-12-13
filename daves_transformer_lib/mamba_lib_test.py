import dataclasses
import typing

from absl.testing import absltest
from absl.testing import parameterized
import tree

import jax
from jax import numpy as jnp
import numpy as np

from flax import linen as nn

from daves_transformer_lib import generate
from daves_transformer_lib import mamba_lib


class SelectiveSSMTest(parameterized.TestCase):

    # Check that running a series of steps produces the same
    # outputs and hiddens as a parallel scan.
    def test_scan_matches_steps(self, D=2, N=3, L=4):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        xs = jax.random.normal(k1, [L, D])
        h0 = jax.random.normal(k3, [D, N])
        ssm = mamba_lib.SelectiveSSM(d_state=N, d_inner=D)
        params = ssm.init(k2, xs)

        # Run the SSM one step at a time.
        h_t = h0
        hs1 = []
        ys1 = []
        for t in range(L):
            h_t, y_t = ssm.apply(params, h_t, xs[t], method=ssm.step)
            hs1.append(h_t)
            ys1.append(y_t)
        hs1, ys1 = jnp.asarray(hs1), jnp.asarray(ys1)

        # Run the SSM with a parallel scan
        ys2, hs2 = ssm.apply(params, xs, initial_hidden=h0)
        print(hs2)
        np.testing.assert_allclose(hs1, hs2, rtol=1e-5)  # type: ignore
        np.testing.assert_allclose(ys1, ys2, rtol=1e-5)

    def test_inititialization(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)

        ssm = mamba_lib.SelectiveSSM(d_state=4, d_inner=2)
        xs = jax.random.normal(k1, [5, 1])
        params = ssm.init(k2, xs)
        np.testing.assert_allclose(params['params']['log_neg_Adiag'],
                                   np.log([[1., 2., 3., 4.], [1., 2., 3., 4.]]))
