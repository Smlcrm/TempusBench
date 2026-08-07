"""Generator: long_memory_fgn.

See eval-synthetic-dataset/synthetic_tasks.md for the design rationale
and the full data-generating process.
"""

from __future__ import annotations

import numpy as np

from ._common import DEFAULT_T, _rng


def long_memory_fgn(T=DEFAULT_T, seed=None, H=0.9):
    """Fractional Gaussian noise with Hurst H=0.9 (long memory).

    Exact simulation via the Hosking/Durbin-Levinson recursion on the fGn
    autocovariance gamma(k) = 0.5*(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}).
    Autocorrelations decay hyperbolically (~ k^{2H-2} = k^{-0.2}); distant
    context remains informative, and predictability decays much more slowly
    with horizon than for any ARMA process.  Scaled by 3 for a working
    amplitude comparable to other tasks.
    """
    rng = _rng(seed)
    k = np.arange(T, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H)
                   + np.abs(k - 1) ** (2 * H))
    x = np.empty(T)
    v = gamma[0]
    x[0] = rng.standard_normal() * np.sqrt(v)
    phi_prev = np.zeros(0)
    for t in range(1, T):
        if t == 1:
            kap = gamma[1] / v
        else:
            kap = (gamma[t] - phi_prev @ gamma[t - 1:0:-1]) / v
        phi_new = np.empty(t)
        phi_new[:t - 1] = phi_prev - kap * phi_prev[::-1]
        phi_new[t - 1] = kap
        v = v * (1.0 - kap ** 2)
        x[t] = phi_new @ x[t - 1::-1] + np.sqrt(v) * rng.standard_normal()
        phi_prev = phi_new
    return 3.0 * x
