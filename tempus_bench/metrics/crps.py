"""
Calculates Continuous Ranked Probability Score.
"""

import numpy as np

from .base_metric import BaseMetric


class CRPS(BaseMetric):

    def __init__(self):
        super().__init__("stochastic")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Approximates the Continuous Ranked Probability Score (CRPS) using the weighted quantile loss approach,
        as described in Park et al. (2022).

        - For each quantile alpha, compute the pinball (quantile) loss evaluated at alpha, weighted by |1 - 2*alpha|.
        - Average the weighted quantile losses over a grid of K quantiles in (0, 1).

        Args:
            y_true: True values, shape (n_timesteps,) or (n_timesteps, num_targets)
            y_pred: Sample predictions, shape (num_samples, n_timesteps) or (num_samples, n_timesteps, num_targets)
            **kwargs: Optional parameters

        Returns:
            Normalized mean CRPS score (averaged across all timesteps and targets).
        """

        # Ensure y_true shape is compatible: (n_timesteps, n_targets)
        if y_true.ndim == 1:
            raise ValueError("y_true must be 2D array, got 1D array")

        S, T, M = y_pred.shape  # S: samples, T: timesteps, M: targets

        # Parameters for quantile averaging
        num_quantiles = 100  # Number of quantiles (exclude 0 and 1)
        quantiles = np.linspace(
            1 / (num_quantiles + 1), num_quantiles / (num_quantiles + 1), num_quantiles
        )  # shape (num_quantiles,)

        # Vectorized computation over quantiles
        # quantiles: (num_quantiles,)
        # y_pred: (S, T, M)
        # Compute quantile predictions for all quantiles at once: (num_quantiles, T, M)
        q_preds = np.quantile(
            y_pred, quantiles[:, None, None], axis=0
        )  # shape (num_quantiles, T, M)

        # Broadcast y_true for computation: shape (num_quantiles, T, M)
        y_true_broadcast = np.broadcast_to(y_true, (num_quantiles, T, M))

        # Compute quantile losses for all quantiles at once: (num_quantiles, T, M)
        diff = y_true_broadcast - q_preds
        q = quantiles[:, None, None]
        quantile_losses = np.maximum(q * diff, (q - 1) * diff)

        # Divide by each alpha per quantile (vectorized): (num_quantiles, T, M)
        weighted_losses = quantile_losses / q

        # Average over all quantiles, time steps, and targets
        crps = np.mean(weighted_losses)

        return crps.item()
