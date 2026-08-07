"""Generator: random_walk.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def random_walk(T=DEFAULT_T, seed=None, sigma=1.0):
    """Driftless random walk: y_t = y_{t-1} + sigma*eps_t, y_0 = 0.

    Difference-stationary (unit root).  Bayes forecast is the last observed
    value at every horizon (martingale), with forecast variance sigma^2 h.
    Models should neither extrapolate local drift nor revert to the
    historical mean.
    """
    rng = _rng(seed)
    return np.cumsum(sigma * rng.standard_normal(T))
