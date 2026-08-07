"""Generator: setar.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def setar(T=DEFAULT_T, seed=None, phi_low=0.95, phi_high=0.4, sigma=0.5):
    """Self-exciting threshold AR with asymmetric persistence (threshold 0):

        y_t = 0.95 y_{t-1} + sigma*eps_t   if y_{t-1} <= 0,
        y_t = 0.40 y_{t-1} + sigma*eps_t   if y_{t-1} >  0.

    Both regimes mean-revert to 0 (geometrically ergodic), but negative
    excursions decay slowly while positive ones die out almost immediately
    - the signature sign-asymmetry of threshold models (cf. unemployment
    dynamics).  A single linear AR fits an averaged phi and systematically
    over-predicts persistence above the threshold and under-predicts it
    below, so the task cleanly separates linear from nonlinear models.
    (An earlier design with opposite-signed regime intercepts was rejected:
    it produced a near period-2 flip-flop that a linear AR with a negative
    lag-1 coefficient imitates well.)
    """
    rng = _rng(seed)
    n = T + BURN_IN
    y = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for t in range(1, n):
        phi = phi_low if y[t - 1] <= 0.0 else phi_high
        y[t] = phi * y[t - 1] + eps[t]
    return y[BURN_IN:]
