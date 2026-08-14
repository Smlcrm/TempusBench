"""Generator: piecewise_linear_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def piecewise_linear_trend(T=DEFAULT_T, seed=None, frac_break=0.6,
                           b1=15.0, b2=-10.0, sigma=1.0):
    """Broken trend: slope b1/T before the break at frac_break*T, b2/T after
    (continuous at the break), plus N(0, sigma^2) noise.

    A slope *reversal* (up then down).  Tests whether a model conditions on
    the post-break regime instead of the full-context average slope.
    """
    rng = _rng(seed)
    t = _t(T)
    tb = frac_break * T
    trend = np.where(t <= tb, b1 * t / T, b1 * tb / T + b2 * (t - tb) / T)
    return 5.0 + trend + sigma * rng.standard_normal(T)
