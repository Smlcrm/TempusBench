"""
Calculates the multivariate Energy Score from forecast samples.

Uses the unbiased S-sample estimator with beta fixed at 1:

    ES = (1/S) sum_i ||s_i - y||_2
         - (1 / (2 S (S-1))) sum_{i != j} ||s_i - s_j||_2

Each sample and the observation are treated as vectors in R^{T*M}
(time and targets flattened). When S == 1 the pairwise term is 0.
"""

import numpy as np

from .base_metric import BaseMetric


class EnergyScore(BaseMetric):
    def __init__(self):
        super().__init__("stochastic")

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the unbiased multivariate Energy Score with beta=1.

        Args:
            y_true: True values, shape (n_timesteps, num_targets).
            y_pred: Forecast samples (preprocessed by base class),
                shape (num_samples, n_timesteps, num_targets).
            **kwargs: Unused; beta is hardcoded to 1.

        Returns:
            Scalar energy score over the full (T, M) forecast vector.
        """
        if y_true.ndim == 1:
            raise ValueError("y_true must be 2D array, got 1D array")
        if y_pred.ndim != 3:
            raise ValueError(
                f"Expected 3D y_pred for energy score, got shape {y_pred.shape}"
            )

        S = y_pred.shape[0]
        flat_s = y_pred.reshape(S, -1)
        flat_y = y_true.reshape(-1)

        term1 = np.mean(np.linalg.norm(flat_s - flat_y, axis=1))
        if S < 2:
            return float(term1)

        diffs = flat_s[:, None, :] - flat_s[None, :, :]
        pair = np.linalg.norm(diffs, axis=-1)
        off = ~np.eye(S, dtype=bool)
        term2 = pair[off].sum() / (2.0 * S * (S - 1))
        return float(term1 - term2)
