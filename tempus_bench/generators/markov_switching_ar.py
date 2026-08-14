"""Generator: markov_switching_ar.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import BURN_IN, DEFAULT_T, _rng


def markov_switching_ar(T=DEFAULT_T, seed=None, p_stay=0.97, sigma=0.5):
    """Recurring regimes: 2-state Markov chain (stay prob 0.97, mean sojourn
    ~33 steps) switching the mean and dynamics of an AR(1):

        state 0:  y_t = -1.5 + 0.5*(y_{t-1}+1.5) + sigma*eps_t
        state 1:  y_t = +1.5 + 0.9*(y_{t-1}-1.5) + sigma*eps_t

    Unlike one-off breaks, regimes recur (~60 switches per series), so the
    regime structure is identifiable from context and the optimal forecast
    is a probability-weighted mixture over future regime paths.  Burn-in
    applied.
    """
    rng = _rng(seed)
    n = T + BURN_IN
    mu = (-1.5, 1.5)
    phi = (0.5, 0.9)
    s = rng.integers(0, 2)
    y = np.zeros(n)
    for t in range(1, n):
        if rng.random() > p_stay:
            s = 1 - s
        y[t] = mu[s] + phi[s] * (y[t - 1] - mu[s]) \
            + sigma * rng.standard_normal()
    return y[BURN_IN:]
