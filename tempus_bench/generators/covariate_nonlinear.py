"""Generator: covariate_nonlinear.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _ar1, _rng


def covariate_nonlinear(T=DEFAULT_T, seed=None, lag=2, sigma_y=0.3):
    """Covariate-driven target with a nonlinear, lagged link (returns
    (T, 2) = [y, x]; x is the covariate, known over context+horizon per the
    TempusBench task definition):

        x_t = 5 sin(2 pi t/24) + AR(1)(0.8, 1.0),
        y_t = 4 tanh(x_{t-lag} / 3) + sigma_y * nu_t.

    Given the covariate path, y is almost deterministic - but only through
    a saturating nonlinearity and a 2-step lag.  Linear covariate handling
    leaves large errors at |x| extremes; ignoring the covariate leaves the
    task nearly unpredictable at long horizons.
    """
    rng = _rng(seed)
    t_full = np.arange(-lag, T, dtype=float)
    x_full = 5.0 * np.sin(2 * np.pi * t_full / PERIOD) \
        + _ar1(rng, T + lag, 0.8, 1.0)
    x = x_full[lag:]
    y = 4.0 * np.tanh(x_full[:T] / 3.0) + sigma_y * rng.standard_normal(T)
    return np.column_stack([y, x])
