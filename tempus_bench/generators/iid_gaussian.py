"""Generator: iid_gaussian.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def iid_gaussian(T=DEFAULT_T, seed=None, mu=0.0, sigma=1.0):
    """White noise: y_t ~ iid N(mu, sigma^2).

    Control task: there is no exploitable structure.  The Bayes point
    forecast is the constant mu and the Bayes MSE is sigma^2 at every
    horizon.  A model that "finds" structure (forecast variance across
    windows well above 0) is hallucinating.
    """
    rng = _rng(seed)
    return mu + sigma * rng.standard_normal(T)
