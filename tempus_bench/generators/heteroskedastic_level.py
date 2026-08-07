"""Generator: heteroskedastic_level.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def heteroskedastic_level(T=DEFAULT_T, seed=None, rel=0.08):
    """Level-dependent (multiplicative-style) heteroskedasticity:

        level_t = 12 + 6 sin(2 pi t / 1024)   (slow, NON-monotonic),
        y_t = level_t + 3 sin(2 pi t/24) + rel*level_t*eps_t.

    Noise sd is proportional to the level.  The level is deliberately
    non-monotonic so that sigma ~ level is distinguishable from sigma ~ t
    (with a monotone level the two are confounded).  Probabilistic metrics
    (CRPS/WIS) reward models whose predictive spread tracks the level.
    """
    rng = _rng(seed)
    t = _t(T)
    level = 12.0 + 6.0 * np.sin(2 * np.pi * t / 1024.0)
    return level + 3.0 * np.sin(2 * np.pi * t / PERIOD) \
        + rel * level * rng.standard_normal(T)
