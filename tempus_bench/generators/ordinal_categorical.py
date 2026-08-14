"""Generator: ordinal_categorical.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _ar1, _rng, _t


def ordinal_categorical(T=DEFAULT_T, seed=None, phi=0.9, sigma=0.4, amp=1.5):
    """Ordered 3-category series: the same latent construction as
    binary_latent_ar, cut at thresholds (-0.8, +0.8):

        y_t = 0 if z_t <= -0.8;  1 if -0.8 < z_t <= 0.8;  2 otherwise.

    Covers the 'categorical' target type of the TempusBench taxonomy with
    an ordered state space (unordered categories have no natural error
    metric under MAE/RMSE-style evaluation).
    """
    rng = _rng(seed)
    z = _ar1(rng, T, phi, sigma) + amp * np.sin(2 * np.pi * _t(T) / PERIOD)
    return np.digitize(z, (-0.8, 0.8)).astype(float)
