"""Generator: poisson_counts.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def poisson_counts(T=DEFAULT_T, seed=None):
    """Non-negative counts, canonical log-link Poisson:

        lambda_t = exp(1.5 + 0.4*(t/T) + 0.7 sin(2 pi t/24)),
        y_t ~ Poisson(lambda_t)     (lambda ranges ~2.2 -> ~13.5).

    Equidispersed (Var = mean).  Discreteness, positivity, and the
    mean-variance link distinguish count data from continuous data; the
    Bayes point forecast is lambda_t (mean) or the Poisson median.
    """
    rng = _rng(seed)
    t = _t(T)
    lam = np.exp(1.5 + 0.4 * (t / T) + 0.7 * np.sin(2 * np.pi * t / PERIOD))
    return rng.poisson(lam).astype(float)
