"""Generator: mean_reverting_ar1.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def mean_reverting_ar1(T=DEFAULT_T, seed=None, phi=0.8, sigma=1.0):
    """Stationary AR(1), phi=0.8: strongly mean-reverting level.

    Bayes forecast decays geometrically to the mean: E[y_{t+h}|y_t] =
    phi^h y_t.  The canonical "stationary movement" task.
    """
    return _ar1(_rng(seed), T, phi, sigma)
