"""Generator: ar2_pseudocyclic.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def ar2_pseudocyclic(T=DEFAULT_T, seed=None, phi1=1.5, phi2=-0.9, sigma=1.0):
    """Stationary AR(2) with complex roots (stochastic pseudo-cycles):

        y_t = phi1*y_{t-1} + phi2*y_{t-2} + sigma*eps_t.

    Spectral peak near period 2 pi / arccos(phi1/(2 sqrt(-phi2))) ~= 9.5,
    but the phase diffuses: unlike true seasonality the cycle cannot be
    extrapolated far ahead.  Distinguishes models that infer dynamics from
    those that pattern-match a fixed calendar period.  Burn-in removes the
    zero-initialisation transient.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y[BURN_IN:]
