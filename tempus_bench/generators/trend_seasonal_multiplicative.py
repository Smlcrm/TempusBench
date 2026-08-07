"""Generator: trend_seasonal_multiplicative.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def trend_seasonal_multiplicative(T=DEFAULT_T, seed=None, sigma_rel=0.05):
    """Non-stationary cyclical, multiplicative composition:

        y_t = trend_t * s_t * (1 + sigma_rel*eps_t),
        trend_t = 10*(1 + t/T),   s_t = 1 + 0.4 sin(2 pi t/24).

    Both the seasonal swing and the noise sd scale with the level (the
    series is strictly positive).  The contrast with the additive task
    isolates whether a model infers the composition type; it is also an
    intrinsically level-heteroskedastic noise task.
    """
    rng = _rng(seed)
    t = _t(T)
    trend = 10.0 * (1.0 + t / T)
    s = 1.0 + 0.4 * np.sin(2 * np.pi * t / PERIOD)
    return trend * s * (1.0 + sigma_rel * rng.standard_normal(T))
