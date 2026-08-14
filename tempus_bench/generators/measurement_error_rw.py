"""Generator: measurement_error_rw.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def measurement_error_rw(T=DEFAULT_T, seed=None, sigma_proc=0.3,
                         sigma_obs=1.0):
    """Local-level state-space model (signal + measurement error):

        x_t = x_{t-1} + sigma_proc * w_t     (latent level, random walk),
        y_t = x_t + sigma_obs * v_t          (noisy observation).

    Signal-to-measurement ratio q = (sigma_proc/sigma_obs)^2 = 0.09.  The
    Bayes forecast is the Kalman-filtered level (equivalently exponential
    smoothing with the steady-state gain ~0.26) - neither the last
    observation (too noisy) nor a long mean (too stale).  This is the
    "measurement error" data-quality task, distinct from process noise.
    """
    rng = _rng(seed)
    x = np.cumsum(sigma_proc * rng.standard_normal(T))
    return x + sigma_obs * rng.standard_normal(T)
