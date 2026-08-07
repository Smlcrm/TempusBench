"""Generator: multi_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, LONG_PERIOD, PERIOD, _rng, _t


def multi_seasonal(T=DEFAULT_T, seed=None, sigma=1.0):
    """Two nested periods (daily-in-weekly analogue):

        y_t = 6 sin(2 pi t/24) + 4 sin(2 pi t/168) + sigma*eps_t.

    168 = 7*24, so the short cycle nests in the long one.  Tests whether a
    model separates superposed periodicities; a single-period model leaves
    the amplitude-4 component (variance 8) unexplained.  For the long cycle
    to matter, use context >= 512 (3 weekly cycles).
    """
    rng = _rng(seed)
    t = _t(T)
    return 6.0 * np.sin(2 * np.pi * t / PERIOD) \
        + 4.0 * np.sin(2 * np.pi * t / LONG_PERIOD) \
        + sigma * rng.standard_normal(T)
