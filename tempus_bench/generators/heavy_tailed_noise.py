"""Generator: heavy_tailed_noise.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng


def heavy_tailed_noise(T=DEFAULT_T, seed=None, df=3.0, scale=1.0):
    """Student-t(3) innovations on the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + scale * t_df / sqrt(df/(df-2)).

    Innovations are standardised to unit variance but have infinite fourth
    moment: rare shocks are ~an order of magnitude larger than Gaussian
    ones.  Tests robustness of context encoding and realistic tail width in
    predictive distributions.
    """
    rng = _rng(seed)
    innov = rng.standard_t(df, size=T) / np.sqrt(df / (df - 2.0))
    return _base_signal(T) + scale * innov
