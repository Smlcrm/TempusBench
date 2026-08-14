"""Generator: mackey_glass.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def mackey_glass(T=DEFAULT_T, seed=None, tau=17.0, beta=0.2, gamma=0.1,
                 n_exp=10, dt=0.1, stride=10):
    """Mackey-Glass delay differential equation (chaotic regime, tau=17):

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t),

    integrated by Euler with step dt=0.1 and sub-sampled every ``stride``
    steps (sampling interval 1 time unit - the standard benchmark setup).
    History initialised at 1.2 plus a small seeded perturbation; the first
    500 time units are discarded as transient.  Smooth quasi-periodic
    chaotic oscillations: strong short-range predictability with slow
    divergence - the classic nonlinear-forecasting benchmark, complementary
    to the logistic map (continuous & smooth vs discrete & jagged).
    Scaled by 10.
    """
    rng = _rng(seed)
    hist = int(round(tau / dt))
    discard_units = 500
    n_steps = hist + (discard_units + T) * stride
    x = np.empty(n_steps)
    x[:hist] = 1.2 + 0.05 * rng.standard_normal()
    for i in range(hist, n_steps):
        xd = x[i - hist]
        x[i] = x[i - 1] + dt * (beta * xd / (1.0 + xd ** n_exp)
                                - gamma * x[i - 1])
    series = x[hist + discard_units * stride::stride][:T]
    return 10.0 * series
