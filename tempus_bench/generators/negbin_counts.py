"""Generator: negbin_counts.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def negbin_counts(T=DEFAULT_T, seed=None, r=3.0):
    """Overdispersed counts via a Gamma-Poisson (negative binomial) mixture
    with the same mean path as poisson_counts:

        mu_t as in poisson_counts;  g_t ~ Gamma(r, mu_t/r);
        y_t ~ Poisson(g_t)   =>   Var = mu_t + mu_t^2 / r.

    With r=3, variance is up to ~5x the Poisson variance at the same mean.
    The paired contrast poisson_counts vs negbin_counts isolates
    overdispersion: point forecasts should coincide, predictive intervals
    must not.
    """
    rng = _rng(seed)
    t = _t(T)
    mu = np.exp(1.5 + 0.4 * (t / T) + 0.7 * np.sin(2 * np.pi * t / PERIOD))
    g = rng.gamma(shape=r, scale=mu / r)
    return rng.poisson(g).astype(float)
