"""Generator: sinusoidal_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def sinusoidal_seasonal(T=DEFAULT_T, seed=None, amp=10.0, period=PERIOD,
                        sigma=1.0):
    """Pure stationary cycle: y_t = amp*sin(2 pi t/period) + sigma*eps_t.

    The reference seasonality task: high SNR, fixed period, sinusoidal
    shape.  Bayes forecast is the sinusoid itself (MSE sigma^2).
    """
    rng = _rng(seed)
    return amp * np.sin(2 * np.pi * _t(T) / period) + sigma * rng.standard_normal(T)
