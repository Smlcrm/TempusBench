"""Generator: random_walk_drift.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def random_walk_drift(T=DEFAULT_T, seed=None, drift=0.08, sigma=1.0):
    """Random walk with drift: y_t = drift + y_{t-1} + sigma*eps_t.

    drift = 0.08 makes the drift identifiable from the realisation the model
    actually sees: the sample-mean increment has s.e. sigma/sqrt(T) ~= 0.022,
    so the drift estimate carries an expected t-statistic of ~3.6 (seed-
    dependent in realisation), and the cumulative
    drift (drift*T ~= 164) dominates the walk's typical excursion
    (sigma*sqrt(T) ~= 45).  (An earlier drift of 0.02 was rejected: it was
    statistically unidentifiable from a single path - s.e. of the estimate
    equal to the drift itself - making the task a duplicate of random_walk.)
    Bayes forecast: last value + drift*h.
    """
    rng = _rng(seed)
    return np.cumsum(drift + sigma * rng.standard_normal(T))
