"""Generator: outlier_contaminated.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _base_signal, _rng


def outlier_contaminated(T=DEFAULT_T, seed=None, p=0.02, out_scale=10.0,
                         sigma=1.0):
    """Additive-outlier contamination of the reference sinusoid:

        y_t = 10 sin(2 pi t/24) + sigma*eps_t + o_t,
        o_t = 0 w.p. 1-p;  +/- U(8, 15)*sigma w.p. p (sign fair-coin).

    Isolated impulses (~2% of points) that carry no information about the
    future.  A robust model ignores them; a fragile context encoding lets a
    single spike distort level/seasonal estimates.
    """
    rng = _rng(seed)
    y = _base_signal(T) + sigma * rng.standard_normal(T)
    mask = rng.random(T) < p
    n_out = int(mask.sum())
    signs = rng.choice([-1.0, 1.0], size=n_out)
    y[mask] += signs * rng.uniform(8.0, 15.0, size=n_out) * sigma
    return y
