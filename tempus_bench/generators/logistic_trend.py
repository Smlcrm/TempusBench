"""Generator: logistic_trend.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng, _t


def logistic_trend(T=DEFAULT_T, seed=None, base=2.0, L=30.0, frac_mid=0.85,
                   frac_width=0.2, sigma=0.5):
    """Sigmoid (saturating) trend: y_t = base + L/(1 + exp(-k(t - t0))) + sigma*eps_t

    with t0 = frac_mid*T and k = 4.4/(frac_width*T) (10%-90% transition
    spans ~frac_width*T steps).  The inflection sits at 0.85*T, so rolling
    evaluation windows successively face acceleration, inflection and
    saturation - the property under test is recognising that apparent
    exponential growth is about to saturate.  The baseline offset keeps the
    series positive before the rise despite additive noise.
    """
    rng = _rng(seed)
    t0 = frac_mid * T
    k = 4.4 / (frac_width * T)
    sig = L / (1.0 + np.exp(-k * (_t(T) - t0)))
    return base + sig + sigma * rng.standard_normal(T)
