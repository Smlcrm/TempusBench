"""Generator: skellam_integer.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def skellam_integer(T=DEFAULT_T, seed=None):
    """Signed integers (difference of two seasonal Poisson flows):

        y_t = N1_t - N2_t,  N1 ~ Poi(exp(1.2 + 0.8 s_t)),
                            N2 ~ Poi(exp(1.2 - 0.8 s_t)),
        s_t = sin(2 pi t/24).

    Anti-phase intensities give a seasonal signed count with mean
    2*exp(1.2)*sinh(0.8 s_t) swinging roughly +/-5.9 - integer-valued but
    crossing zero, covering the 'count (negative and positive)' target
    type that pure Poisson tasks cannot.
    """
    rng = _rng(seed)
    s = np.sin(2 * np.pi * _t(T) / PERIOD)
    n1 = rng.poisson(np.exp(1.2 + 0.8 * s))
    n2 = rng.poisson(np.exp(1.2 - 0.8 * s))
    return (n1 - n2).astype(float)
