"""Generator: intermittent_demand.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def intermittent_demand(T=DEFAULT_T, seed=None):
    """Zero-inflated intermittent demand (Croston setting):

        occurrence: o_t ~ Bernoulli(p_t),
                    p_t = sigmoid(-1.2 + 0.8 sin(2 pi t/24))  (~0.12-0.4),
        size:       s_t ~ 1 + Poisson(3),
        y_t = o_t * s_t.

    ~75% zeros with seasonally varying occurrence odds.  Per-step squared
    error is minimised by p_t*E[s] (a fractional value!), not by the modal
    zero - the classic intermittent-demand trap; also probes zero-inflation
    handling in probabilistic outputs.
    """
    rng = _rng(seed)
    t = _t(T)
    p = 1.0 / (1.0 + np.exp(-(-1.2 + 0.8 * np.sin(2 * np.pi * t / PERIOD))))
    occ = rng.random(T) < p
    size = 1 + rng.poisson(3.0, size=T)
    return (occ * size).astype(float)
