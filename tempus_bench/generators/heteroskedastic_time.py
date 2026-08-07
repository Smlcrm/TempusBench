"""Generator: heteroskedastic_time.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng, _t


def heteroskedastic_time(T=DEFAULT_T, seed=None, sigma0=0.5, sigma1=2.5):
    """Time-driven variance growth: y_t = 10 sin(2 pi t/24) + sigma(t)*eps_t,
    with sigma(t) = sigma0 + (sigma1-sigma0)*(t/T) rising 0.5 -> 2.5.

    Conditional-mean structure is unchanged from the reference sinusoid;
    only the noise level trends.  Point metrics should degrade gracefully
    and predictive intervals should widen with t.
    """
    rng = _rng(seed)
    sig = sigma0 + (sigma1 - sigma0) * (_t(T) / T)
    return _base_signal(T) + sig * rng.standard_normal(T)
