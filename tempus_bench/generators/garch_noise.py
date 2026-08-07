"""Generator: garch_noise.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def garch_noise(T=DEFAULT_T, seed=None, omega=0.05, alpha=0.1, beta=0.85):
    """GARCH(1,1) with zero conditional mean (volatility clustering):

        y_t = sigma_t * eps_t,   sigma_t^2 = omega + alpha*y_{t-1}^2
                                             + beta*sigma_{t-1}^2.

    alpha+beta = 0.95 < 1 (covariance-stationary, unconditional variance
    omega/(1-alpha-beta) = 1); initialised at the unconditional variance
    with a burn-in.  The Bayes point forecast is 0 - the task is purely
    probabilistic: predictive spread must expand and contract with the
    volatility state.  Excess kurtosis is the visible fingerprint.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.empty(n)
    var = omega / (1.0 - alpha - beta)
    eps = rng.standard_normal(n)
    y[0] = np.sqrt(var) * eps[0]
    for t in range(1, n):
        var = omega + alpha * y[t - 1] ** 2 + beta * var
        y[t] = np.sqrt(var) * eps[t]
    return y[BURN_IN:]
