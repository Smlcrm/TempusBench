"""Generator: exponential_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def exponential_trend(T=DEFAULT_T, seed=None, y0=3.0, growth=10.0, sigma=0.8):
    """Exponential growth: y_t = y0 * growth^(t/T) + sigma*eps_t.

    Level rises 3 -> 30.  Tests out-of-range level extrapolation with
    accelerating increments; linear extrapolation of the context under-
    shoots the horizon.
    """
    rng = _rng(seed)
    return y0 * np.power(growth, _t(T) / T) + sigma * rng.standard_normal(T)
