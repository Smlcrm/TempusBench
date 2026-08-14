"""Generator: mv_var.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def mv_var(T=DEFAULT_T, seed=None, sigma=0.5):
    """Bivariate VAR(1) with genuine cross-dynamics (returns (T, 2)):

        x_t = A x_{t-1} + e_t,  A = [[0.7, 0.25], [-0.2, 0.6]],
        e_t ~ N(0, sigma^2 I).

    Eigenvalues of A are complex with modulus ~0.69 (stationary, damped
    rotational dynamics).  Each series Granger-causes the other, so the
    optimal forecast of either series requires both histories - the
    canonical test that a multivariate model actually uses cross-series
    lags.
    """
    rng = _rng(seed)
    A = np.array([[0.7, 0.25], [-0.2, 0.6]])
    x = np.zeros((T + BURN_IN, 2))
    e = sigma * rng.standard_normal((T + BURN_IN, 2))
    for t in range(1, T + BURN_IN):
        x[t] = A @ x[t - 1] + e[t]
    return x[BURN_IN:]
