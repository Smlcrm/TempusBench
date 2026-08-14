"""Generator: lumpy_demand.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def lumpy_demand(T=DEFAULT_T, seed=None, p=0.08, mu=1.0, s=1.0):
    """Lumpy demand (Syntetos-Boylan 'lumpy' quadrant): rare occurrences
    AND highly variable sizes:

        o_t ~ Bernoulli(0.08) i.i.d.  (average inter-demand interval 12.5),
        s_t = ceil(LogNormal(mu=1.0, sigma=1.0))  (heavy-tailed, CV^2 ~ 1.7),
        y_t = o_t * s_t.

    ~92% zeros with sizes ranging from 1 to occasional ~40-unit spikes.
    Complements intermittent_demand (frequent-ish occurrences, low size
    variability): here BOTH the timing and the magnitude are hard, the
    regime where point forecasts are least informative and probabilistic
    (quantile) forecasts carry all the value.  A 512-step context still
    contains ~41 demand events, so the size distribution is estimable.
    """
    rng = _rng(seed)
    occ = rng.random(T) < p
    size = np.ceil(rng.lognormal(mean=mu, sigma=s, size=T))
    return (occ * size).astype(float)
