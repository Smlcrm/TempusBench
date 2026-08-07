"""Generator: logistic_map.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def logistic_map(T=DEFAULT_T, seed=None, r=3.9, obs_sigma=0.0):
    """Chaotic logistic map: x_{t+1} = r x_t (1 - x_t), r=3.9, x in (0,1).

        y_t = 10*x_t + obs_sigma*eps_t   (scaled to working amplitude).

    Fully deterministic: one-step dynamics are exactly learnable (a smooth
    quadratic map), but sensitivity to initial conditions (Lyapunov exponent
    ~0.5) makes long horizons intrinsically unpredictable - point accuracy
    must degrade toward the invariant distribution at a known exponential
    rate.  Tests short-horizon nonlinear dynamics *and* long-horizon
    uncertainty growth.  Initial condition drawn U(0.1, 0.9); 200-step
    transient discarded.
    """
    rng = _rng(seed)
    x = rng.uniform(0.1, 0.9)
    for _ in range(200):
        x = r * x * (1.0 - x)
    out = np.empty(T)
    for t in range(T):
        x = r * x * (1.0 - x)
        out[t] = x
    y = 10.0 * out
    if obs_sigma > 0:
        y = y + obs_sigma * rng.standard_normal(T)
    return y
