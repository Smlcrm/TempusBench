"""Generator: evolving_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _ar1, _rng, _t


def evolving_seasonal(T=DEFAULT_T, seed=None, sigma=1.0):
    """Slowly morphing seasonal shape via drifting Fourier coefficients.

        y_t = sum_{k=1..3} [a_k(t) cos(k w t) + b_k(t) sin(k w t)] + sigma*eps_t

    with w = 2 pi/24.  Each coefficient follows a highly persistent AR(1)
    (phi = 0.997, innovation sd 0.15) around base values a = (0,0,0),
    b = (6,3,1.5) - mean-reverting, so amplitudes stay bounded (unlike a
    random-walk drift, which would wander arbitrarily).  The seasonal shape
    a model should use is the *recent* one, not the context-wide average.
    """
    rng = _rng(seed)
    t = _t(T)
    w = 2 * np.pi / PERIOD
    base_a = np.array([0.0, 0.0, 0.0])
    base_b = np.array([6.0, 3.0, 1.5])
    y = sigma * rng.standard_normal(T)
    for k in range(3):
        a = _ar1(rng, T, 0.997, 0.15, mu=base_a[k])
        b = _ar1(rng, T, 0.997, 0.15, mu=base_b[k])
        y += a * np.cos((k + 1) * w * t) + b * np.sin((k + 1) * w * t)
    return y
