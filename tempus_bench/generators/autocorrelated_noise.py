"""Generator: autocorrelated_noise.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _base_signal, _rng


def autocorrelated_noise(T=DEFAULT_T, seed=None, phi=0.8, sigma=0.6):
    """AR(1) errors around a deterministic sinusoid:

        y_t = 10 sin(2 pi t/24) + u_t,   u_t = phi*u_{t-1} + sigma*eps_t.

    The optimal forecast corrects the sinusoid by phi^h times the last
    residual - a model that treats residuals as white noise leaves
    first-lag structure on the table.  (Classic regression-with-ARMA-errors
    setting.)
    """
    rng = _rng(seed)
    return _base_signal(T) + _ar1(rng, T, phi, sigma)
