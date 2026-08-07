"""Generator: irregular_period_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def irregular_period_seasonal(T=DEFAULT_T, seed=None, amp=10.0, sigma=1.0,
                              mod_depth=0.3, mod_period=512.0):
    """Frequency-modulated cycle with smoothly drifting period.

    The instantaneous frequency f(t) = (1/24)*(1 + mod_depth*sin(2 pi t /
    mod_period)) is integrated into a phase (phi_t = 2 pi * cumsum f), and

        y_t = amp * sin(phi_t) + sigma*eps_t.

    The local period therefore truly varies between ~18.5 and ~34 steps.
    (Naively writing sin(2 pi t / p(t)) does NOT do this - the derivative
    of t/p(t) is not 1/p(t) - hence the phase-accumulation construction.)
    Tests phase tracking when the period cannot be assumed constant.
    """
    rng = _rng(seed)
    t = _t(T)
    freq = (1.0 / PERIOD) * (1.0 + mod_depth * np.sin(2 * np.pi * t / mod_period))
    phase = 2 * np.pi * np.cumsum(freq)
    return amp * np.sin(phase) + sigma * rng.standard_normal(T)
