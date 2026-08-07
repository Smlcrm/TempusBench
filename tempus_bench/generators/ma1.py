"""Generator: ma1.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def ma1(T=DEFAULT_T, seed=None, theta=0.8, sigma=1.0):
    """Invertible MA(1): y_t = eps_t + theta*eps_{t-1}, eps ~ N(0, sigma^2).

    Memory of exactly one lag: the Bayes forecast is non-trivial at horizon
    1 (E[y_{t+1}|F_t] = theta*eps_t) and exactly 0 for horizons >= 2.  Tests
    both short-memory exploitation and the calibration to *stop* predicting
    beyond the memory of the process.
    """
    rng = _rng(seed)
    eps = rng.standard_normal(T + 1) * sigma
    return eps[1:] + theta * eps[:-1]
