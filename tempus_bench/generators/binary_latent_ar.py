"""Generator: binary_latent_ar.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _ar1, _rng, _t


def binary_latent_ar(T=DEFAULT_T, seed=None, phi=0.9, sigma=0.4, amp=1.5):
    """Binary series from a thresholded latent process:

        z_t = phi*z_{t-1} + sigma*eps_t + amp*sin(2 pi t/24) applied as
        z_t = AR(1)(phi, sigma) + amp*sin(2 pi t/24),
        y_t = 1{z_t > 0}  in {0, 1}.

    Persistence and seasonality make P(y_{t+h}=1 | context) genuinely
    dynamic (roughly 0.1-0.9 over a cycle); the Bayes probability is the
    Gaussian CDF of the predicted latent mean over its predictive sd.
    Tests forecasting on a two-point support where regression-style outputs
    must be interpretable as probabilities/thresholded states.
    """
    rng = _rng(seed)
    z = _ar1(rng, T, phi, sigma) + amp * np.sin(2 * np.pi * _t(T) / PERIOD)
    return (z > 0.0).astype(float)
