"""Generator: mv_leadlag.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def mv_leadlag(T=DEFAULT_T, seed=None, lag=8, beta=1.0, sigma_y=0.3,
               phi=0.95, sigma_x=1.0):
    """Leading indicator: x leads y by ``lag`` steps (returns (T, 2) =
    [y, x]):

        x_t = 0.95 x_{t-1} + eps_t,      y_t = beta * x_{t-lag} + 0.3 nu_t.

    For horizons h <= lag, y_{t+h} is (up to small noise) *already visible*
    in x's context - the multivariate Bayes MSE is 0.09 while the best
    univariate forecast inherits x's innovation variance.  The purest test
    that a model exploits cross-series lead-lag information; with h > lag
    the advantage decays, so keep lag >= h/2 relative to the task horizon.
    """
    rng = _rng(seed)
    x_full = _ar1(rng, T + lag, phi, sigma_x)
    x = x_full[lag:]
    y = beta * x_full[:T] + sigma_y * rng.standard_normal(T)
    return np.column_stack([y, x])
