"""Generator: trend_seasonal_additive.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def trend_seasonal_additive(T=DEFAULT_T, seed=None, sigma=1.0):
    """Non-stationary cyclical, additive composition:

        y_t = 10 + 15*(t/T) + 8 sin(2 pi t/24) + sigma*eps_t.

    Seasonal amplitude is constant while the level rises - the additive
    benchmark, and simultaneously the homoskedastic reference case for the
    noise category (constant conditional variance sigma^2).
    """
    rng = _rng(seed)
    t = _t(T)
    return 10.0 + 15.0 * (t / T) + 8.0 * np.sin(2 * np.pi * t / PERIOD) \
        + sigma * rng.standard_normal(T)
