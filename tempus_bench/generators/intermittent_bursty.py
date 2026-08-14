"""Generator: intermittent_bursty.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def intermittent_bursty(T=DEFAULT_T, seed=None, phi=0.95, thresh=0.84):
    """Intermittent demand with serially dependent occurrence:

        z_t = 0.95 z_{t-1} + sqrt(1-0.95^2) eps_t   (latent, stationary sd 1),
        o_t = 1{z_t > 0.84}   (marginal occurrence prob ~0.2, in runs),
        y_t = o_t * (1 + Poisson(2)).

    The persistent latent makes demand arrive in bursts separated by long
    quiet spells - the occurrence indicator has lag-1 autocorrelation
    ~0.5 instead of Croston's i.i.d.-interval assumption.  The optimal
    forecast is state-dependent: P(demand next step) is high right after
    observed demand and decays through a quiet spell.  Models that estimate
    a single average demand rate are systematically wrong in both states.
    """
    rng = _rng(seed)
    sigma = np.sqrt(1.0 - phi**2)
    z = _ar1(rng, T, phi, sigma)
    occ = z > thresh
    size = 1 + rng.poisson(2.0, size=T)
    return (occ * size).astype(float)
