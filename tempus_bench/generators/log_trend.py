"""Generator: log_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def log_trend(T=DEFAULT_T, seed=None, base=2.0, c=2.0, sigma=0.5):
    """Logarithmic (decelerating) trend: y_t = base + c*ln(1+t) + sigma*eps_t.

    Growth without a finite asymptote but with ever-slowing increments;
    naive linear extrapolation overshoots.  The baseline offset keeps the
    series positive at early t despite additive noise.
    """
    rng = _rng(seed)
    return base + c * np.log1p(_t(T)) + sigma * rng.standard_normal(T)
