"""Generator: level_shift.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng, _t


def level_shift(T=DEFAULT_T, seed=None, frac_break=0.7, shift=6.0, sigma=1.0):
    """One-off mean shift on the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + shift*1{t >= 0.7T} + sigma*eps_t.

    A permanent +6 sigma level jump.  Post-break, the correct level is the
    new one; models anchored to the full-context mean systematically bias
    low.  (Under rolling windows, early windows also probe behaviour when
    the break sits inside the forecast context at varying depths.)
    """
    rng = _rng(seed)
    t = _t(T)
    return _base_signal(T) + shift * (t >= frac_break * T) \
        + sigma * rng.standard_normal(T)
