"""Shared constants and helpers for the data generators.

Extracted verbatim from the original synthetic_generators.py so the
per-generator modules stay byte-identical in behaviour.
"""

from __future__ import annotations

import numpy as np

BURN_IN = 512             # burn-in steps for recursive processes
DEFAULT_T = 2048          # default series length
LONG_PERIOD = 168         # long seasonal period ("weekly" analogue)
PERIOD = 24               # base seasonal period ("daily" analogue)


def _ar1(rng, T, phi, sigma, mu=0.0):
    """Stationary AR(1): x_t = mu + phi*(x_{t-1}-mu) + sigma*eps_t.

    Initialised from the stationary distribution N(mu, sigma^2/(1-phi^2)),
    so no burn-in is needed.  Requires |phi| < 1.
    """
    x = np.empty(T)
    x[0] = mu + rng.standard_normal() * sigma / np.sqrt(1.0 - phi**2)
    eps = rng.standard_normal(T) * sigma
    for t in range(1, T):
        x[t] = mu + phi * (x[t - 1] - mu) + eps[t]
    return x


def _base_signal(T):
    """Common carrier signal for the noise-category tasks."""
    return 10.0 * np.sin(2 * np.pi * _t(T) / PERIOD)


def _rng(seed):
    return np.random.default_rng(seed)


def _t(T):
    """Time index 0, 1, ..., T-1 as float."""
    return np.arange(T, dtype=float)
