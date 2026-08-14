"""Generator: mv_common_factor.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _ar1, _rng


def mv_common_factor(T=DEFAULT_T, seed=None, m=4):
    """Factor structure: m=4 series driven by one latent AR(1) factor
    (returns (T, 4)):

        f_t = 0.9 f_{t-1} + eps_t;   y_it = loading_i * f_t + u_it,
        loadings = (1.0, 0.8, -0.6, 0.5);  u_it = AR(1)(0.4, 1.0) idiosync.

    The common factor carries ~70% of total panel variance - dominant but
    not trivially so.  Cross-sectional averaging de-noises the factor: the
    panel reveals f_t more precisely than any single series does, so pooled
    forecasts strictly beat per-series univariate ones.  Tests factor
    extraction / cross-sectional information pooling.
    """
    rng = _rng(seed)
    loadings = np.array([1.0, 0.8, -0.6, 0.5])
    f = _ar1(rng, T, 0.9, 1.0)
    out = np.empty((T, m))
    for i in range(m):
        out[:, i] = loadings[i] * f + _ar1(rng, T, 0.4, 1.0)
    return out
