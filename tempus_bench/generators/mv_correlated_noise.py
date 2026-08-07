"""Generator: mv_correlated_noise.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def mv_correlated_noise(T=DEFAULT_T, seed=None, m=3, phi=0.7, rho=0.8):
    """m AR(1) series with strongly correlated innovations (returns (T, m)):

        x_t = phi * x_{t-1} + e_t,   e_t ~ N(0, Sigma),
        Sigma_ij = rho^{|i != j|}   (equicorrelation 0.8).

    Because the transition is diagonal, the conditional MEAN of each series
    given the joint past equals its univariate forecast - contemporaneous
    correlation adds nothing to point accuracy.  What it does change is the
    joint predictive distribution.  This task therefore tests joint/
    probabilistic calibration specifically; a gap between a model's
    univariate and multivariate point scores here signals confusion, not
    skill.
    """
    rng = _rng(seed)
    cov = np.full((m, m), rho) + (1.0 - rho) * np.eye(m)
    L = np.linalg.cholesky(cov)
    e = rng.standard_normal((T + BURN_IN, m)) @ L.T
    x = np.zeros((T + BURN_IN, m))
    for t in range(1, T + BURN_IN):
        x[t] = phi * x[t - 1] + e[t]
    return x[BURN_IN:]
