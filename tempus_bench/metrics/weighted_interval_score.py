"""
Calculates the Weighted Interval Score (WIS) for deciles.

The Weighted Interval Score is a proper scoring rule for probabilistic forecasts
that evaluates prediction intervals at multiple confidence levels. It combines
sharpness (interval width) and calibration (coverage) into a single score.

Mathematical formula:
WIS_A = Σ_{α∈A} α * IS_α([l_α, u_α], y)

Where:
- IS_α = (u_α - l_α) + (2/α) * dist(y, [l_α, u_α])
- l_α = F⁻¹(α/2) and u_α = F⁻¹(1 - α/2) are the quantiles from samples
- dist(y, [l_α, u_α]) = max(0, max(l_α - y, y - u_α)) is the distance penalty
- α is the significance level for each prediction interval

The score rewards narrow intervals (sharpness) and penalizes when the true value
falls outside the prediction interval (calibration).
"""

import numpy as np

from .base_metric import BaseMetric


class WeightedIntervalScore(BaseMetric):
    def __init__(self):
        super().__init__("stochastic")

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, num_quantiles: int = 100, **kwargs
    ) -> float:
        """
        Computes Weighted Interval Score (WIS) for deciles.

        Args:
            y_true: True values, shape (n_timesteps, 1) or (n_timesteps, num_targets)
            y_pred: Prediction samples, shape (num_samples, forecast_horizon, num_targets)
            **kwargs: Optional parameters including 'num_quantiles' for the number of intervals to compute

        Returns:
            float: WIS score averaged across all timesteps and targets

        Raises:
            ValueError: If task_type is not 'stochastic'
        """

        alphas = np.linspace(0.0, 1.0, num=num_quantiles)  # (num_quantiles,)
        # Exclude alpha=0 (interval width 0) for numerical stability (avoid 2/0 in IS_a)
        valid = alphas > 0
        alphas = alphas[
            valid
        ]  # shape (num_quantiles',) where num_quantiles' = number of nonzero alpha

        l = np.quantile(
            y_pred, alphas[:, None, None] / 2, axis=0
        )  # shape (num_quantiles', H, M)
        u = np.quantile(
            y_pred, 1 - alphas[:, None, None] / 2, axis=0
        )  # shape (num_quantiles', H, M)

        y_true_b = np.broadcast_to(y_true, l.shape)
        dist = np.maximum(
            0, np.maximum(l - y_true_b, y_true_b - u)
        )  # shape (num_quantiles', H, M)

        interval_scores = (u - l) + (
            2.0 / alphas[:, None, None]
        ) * dist  # shape (num_quantiles', H, M)
        weighted_interval_scores = np.sum(
            alphas[:, None, None] * interval_scores, axis=0
        )

        average_weighted_interval_scores = np.mean(weighted_interval_scores)

        return average_weighted_interval_scores.item()
