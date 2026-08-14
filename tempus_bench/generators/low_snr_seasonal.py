"""Generator: low_snr_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def low_snr_seasonal(T=DEFAULT_T, seed=None, amp=1.0, sigma=2.0):
    """Weak sinusoid buried in noise: y_t = amp*sin(2 pi t/24) + sigma*eps_t.

    Signal-to-noise ratio (variance) = amp^2/2 / sigma^2 = 0.125.  With ~85
    full cycles in the context the seasonal component is statistically
    recoverable (periodogram peak >> noise floor), so a good model should
    beat the unconditional mean; a model that gives up and predicts the
    mean loses amp^2/2 of explainable variance.
    """
    rng = _rng(seed)
    t = _t(T)
    return amp * np.sin(2 * np.pi * t / PERIOD) + sigma * rng.standard_normal(T)
