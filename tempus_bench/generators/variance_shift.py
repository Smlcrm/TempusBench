"""Generator: variance_shift.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng, _t


def variance_shift(T=DEFAULT_T, seed=None, frac_break=0.6, sigma1=1.0,
                   sigma2=2.5):
    """One-off variance break: sigma jumps 1.0 -> 2.5 at 0.6T on the
    reference sinusoid (conditional mean unchanged).

    Point-forecast difficulty is unchanged; predictive-interval width must
    change.  Complements level_shift by breaking the second moment only.
    """
    rng = _rng(seed)
    t = _t(T)
    sig = np.where(t < frac_break * T, sigma1, sigma2)
    return _base_signal(T) + sig * rng.standard_normal(T)
