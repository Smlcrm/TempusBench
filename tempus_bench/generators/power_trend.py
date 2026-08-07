"""Generator: power_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def power_trend(T=DEFAULT_T, seed=None, base=2.0, scale=10.0, p=0.5,
                sigma=0.5):
    """Power-law ("p-root") trend: y_t = base + scale*(t/T)^p + sigma*eps_t.

    With p=0.5, concave sub-linear growth; increments decay like t^(p-1).
    Intermediate between linear and logarithmic deceleration.  The baseline
    offset keeps the series positive at early t despite additive noise.
    """
    rng = _rng(seed)
    return base + scale * np.power(_t(T) / T, p) + sigma * rng.standard_normal(T)
