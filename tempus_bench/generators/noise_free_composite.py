"""Generator: noise_free_composite.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _t


def noise_free_composite(T=DEFAULT_T, seed=None):
    """Deterministic, noise-free signal: trend + two incommensurate sines.

        y_t = 0.004 t + 6 sin(2 pi t / 24) + 3 sin(2 pi t / 41)

    (24 and 41 are coprime, so the joint pattern repeats only every 984
    steps.)  The Bayes error is exactly zero: this measures a model's
    precision ceiling and its ability to represent superposed periodicities
    without the excuse of noise.  ``seed`` is accepted for interface
    uniformity but unused.
    """
    t = _t(T)
    return 0.004 * t + 6.0 * np.sin(2 * np.pi * t / PERIOD) \
        + 3.0 * np.sin(2 * np.pi * t / 41.0)
