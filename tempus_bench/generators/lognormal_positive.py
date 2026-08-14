"""Generator: lognormal_positive.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _ar1, _rng, _t


def lognormal_positive(T=DEFAULT_T, seed=None):
    """Strictly positive continuous series, multiplicative on every scale:

        log y_t = 2 + 0.5*(t/T) + 0.4 sin(2 pi t/24) + u_t,
        u_t = AR(1)(phi=0.7, sigma=0.2).

    Level ~7 -> ~12 with conditionally log-normal noise: sd proportional to
    level and right-skewed marginals.  The natural model is additive in
    logs; tests whether a forecaster respects positivity (no negative
    samples/intervals) and multiplicative error structure.
    """
    rng = _rng(seed)
    t = _t(T)
    logy = 2.0 + 0.5 * (t / T) + 0.4 * np.sin(2 * np.pi * t / PERIOD) \
        + _ar1(rng, T, 0.7, 0.2)
    return np.exp(logy)
