"""
Calculates Quantile Score (QS) for deciles using the tilted ℓ₁ loss.
"""

import numpy as np

from .base_metric import BaseMetric


class QuantileScore(BaseMetric):
    def __init__(self):
        super().__init__("stochastic")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the Quantile Score (QS) for deciles using the tilted ℓ₁ loss.

        Args:
            y_true: True observed values, shape (num_steps, 1) or (num_steps, num_targets)
            y_pred: Forecast samples, shape (num_samples, num_steps, num_targets)
            **kwargs: Requires 'num_quantiles' parameter

        Returns:
            float: The mean Quantile Score across all timesteps and targets.
        """
        num_quantiles = kwargs["num_quantiles"]
        taus = np.linspace(0.0, 1.0, num=num_quantiles)
        # Compute quantile predictions for all taus at once: (num_quantiles, num_steps, num_targets)
        q_taus = np.quantile(y_pred, taus, axis=0)
        # Broadcast shapes for vectorized computation
        # y_true:        (num_steps, num_targets) -> (num_quantiles, num_steps, num_targets)
        y_true_b = np.broadcast_to(y_true, q_taus.shape)
        # Compute tau and (tau-1) for broadcast
        taus_b = taus[:, None, None]
        # Vectorized 'rho'
        u = y_true_b - q_taus
        quantile_scores = np.sum(np.maximum(taus_b * u, (taus_b - 1) * u), axis=0)

        # Aggregate across quantiles
        quantile_score = np.mean(quantile_scores)

        return quantile_score.item()
