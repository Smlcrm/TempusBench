"""Generator: near_unit_root_ar1.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def near_unit_root_ar1(T=DEFAULT_T, seed=None, phi=0.995, sigma=1.0):
    """AR(1) with phi=0.995: stationary but nearly integrated.

    Discriminates models that over-difference (treat as random walk => flat
    forecasts, ignoring slow reversion) from those that over-mean-revert.
    Stationary sd = sigma/sqrt(1-phi^2) ~= 10; mean-reversion half-life
    ln(0.5)/ln(phi) ~= 138 steps, i.e. reversion is visible across a
    512-step context but negligible within a 64-step horizon.
    """
    return _ar1(_rng(seed), T, phi, sigma)
