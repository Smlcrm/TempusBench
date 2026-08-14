"""Generator: skewed_noise.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng


def skewed_noise(T=DEFAULT_T, seed=None, s=0.8):
    """Right-skewed innovations (centred, unit-variance log-normal):

        e_t = (LN(0, s^2) - exp(s^2/2)) / sqrt((exp(s^2)-1) exp(s^2)),
        y_t = 10 sin(2 pi t/24) + e_t.

    Mean-zero but skew ~3.7: the median lies below the mean, so a model
    trained toward the median (MAE-style) biases low on the mean and vice
    versa.  Separates mean- from median-calibrated forecasters.
    """
    rng = _rng(seed)
    raw = rng.lognormal(mean=0.0, sigma=s, size=T)
    m = np.exp(s**2 / 2.0)
    sd = np.sqrt((np.exp(s**2) - 1.0) * np.exp(s**2))
    return _base_signal(T) + (raw - m) / sd
