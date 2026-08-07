"""Generator: zero_inflated_continuous.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def zero_inflated_continuous(T=DEFAULT_T, seed=None):
    """Zero-inflated *continuous* series (precipitation analogue):

        p_t = sigmoid(-0.8 + 1.2 sin(2 pi t/24))  (occurrence prob ~0.12-0.6),
        a_t ~ LogNormal(0.8, 0.5)                 (positive continuous amount),
        y_t = 1{u_t < p_t} * a_t.

    A mixed discrete-continuous marginal: an atom at exactly 0 plus a
    right-skewed density on (0, inf), with seasonally varying occurrence
    odds.  Unlike the count-valued intermittent tasks, the nonzero part is
    continuous, so quantization tricks do not apply; predictive
    distributions must represent both the zero mass and the continuous
    tail (as in precipitation forecasting).
    """
    rng = _rng(seed)
    t = _t(T)
    p = 1.0 / (1.0 + np.exp(-(-0.8 + 1.2 * np.sin(2 * np.pi * t / PERIOD))))
    occ = rng.random(T) < p
    amount = rng.lognormal(mean=0.8, sigma=0.5, size=T)
    return (occ * amount).astype(float)
