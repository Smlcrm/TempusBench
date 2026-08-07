"""Generator: mv_cointegrated.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def mv_cointegrated(T=DEFAULT_T, seed=None):
    """Cointegrated pair sharing one stochastic trend (returns (T, 2)):

        w_t = w_{t-1} + eps_t (random walk);
        y1_t = w_t + u1_t,   y2_t = 0.7 w_t + u2_t,
        u_i = AR(1)(0.5, 0.5) stationary.

    Each series alone is a unit-root process, but the spread y1 - y2/0.7
    is stationary: deviations between the series are temporary and
    forecastably close.  Tests whether a multivariate model exploits the
    error-correction structure (forecasting the *pair* coherently) instead
    of forecasting two independent random walks.
    """
    rng = _rng(seed)
    w = np.cumsum(rng.standard_normal(T))
    y1 = w + _ar1(rng, T, 0.5, 0.5)
    y2 = 0.7 * w + _ar1(rng, T, 0.5, 0.5)
    return np.column_stack([y1, y2])
