"""Generator: nonsinusoidal_seasonal.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, PERIOD, _rng, _t


def nonsinusoidal_seasonal(T=DEFAULT_T, seed=None, kappa=2.0, sigma=1.0):
    """Sharp, asymmetric periodic shape (exponentiated-sine "spike train"):

        s(t) = exp(kappa*[sin(w t) + 0.5 sin(2 w t)]),  w = 2 pi / 24,
        y_t  = 10 * (s(t) - mean(s)) / (max(s) - min(s)) * 2 + sigma*eps_t.

    The pattern is peaked and left-right asymmetric - deliberately far from
    a sinusoid - normalised to amplitude ~10.  Tests seasonal *shape*
    fidelity: a Fourier-truncated or smoothness-biased model rounds off the
    peaks, which the per-step metrics punish.
    """
    rng = _rng(seed)
    t = _t(T)
    w = 2 * np.pi / PERIOD
    s = np.exp(kappa * (np.sin(w * t) + 0.5 * np.sin(2 * w * t)))
    s = (s - s.mean()) / (s.max() - s.min()) * 20.0
    return s + sigma * rng.standard_normal(T)
