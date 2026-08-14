"""Generator: linear_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def linear_trend(T=DEFAULT_T, seed=None, a=10.0, b=20.0, sigma=1.0):
    """Trend-stationary linear trend: y_t = a + b*(t/T) + sigma*eps_t.

    Deterministic rise of 20 = 20 noise sds over the series.  Unlike the
    random walk with drift, forecast uncertainty does NOT grow with horizon
    (Bayes MSE = sigma^2 for all h) - the diagnostic pair for
    trend- vs difference-stationarity.
    """
    rng = _rng(seed)
    return a + b * (_t(T) / T) + sigma * rng.standard_normal(T)
