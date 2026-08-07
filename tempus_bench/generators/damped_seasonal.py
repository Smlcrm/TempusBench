"""Generator: damped_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def damped_seasonal(T=DEFAULT_T, seed=None, amp0=12.0, sigma=1.0):
    """Regressive (decaying-amplitude) cycle:

        y_t = amp0 * exp(-t/tau) * sin(2 pi t/24) + sigma*eps_t,  tau = T/2.

    Amplitude decays 12 -> 12*e^-2 ~= 1.6 across the series.  The model
    must extrapolate the *envelope*, not the last-seen amplitude: repeating
    the final context cycle overshoots the horizon amplitude.
    """
    rng = _rng(seed)
    t = _t(T)
    return amp0 * np.exp(-t / (T / 2.0)) * np.sin(2 * np.pi * t / PERIOD) \
        + sigma * rng.standard_normal(T)
