"""
Calculates Quantile Score (QS) for deciles using the tilted ℓ₁ loss.
"""
import numpy as np

from .base_metric import BaseMetric


class QuantileScore(BaseMetric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the Quantile Score (QS) for deciles using the tilted ℓ₁ loss.

        Args:
            y_true: True observed values, shape (num_steps, 1) or (num_steps, num_targets)
            y_pred: Forecast samples, shape (num_samples, num_steps, num_targets)
            **kwargs: Requires 'task_type'='stochastic'

        Returns:
            float: The mean Quantile Score across all timesteps and targets.

        Raises:
            ValueError: If 'task_type' is not 'stochastic'
        """
        y_ppred = self.process_y_pred(y_true, y_pred, **kwargs)
        rho = lambda u, tau: np.maximum(tau * u, (tau - 1) * u)
        QS = np.zeros(y_true.shape)
        for tau in np.arange(0.0, 1.1, 0.1):
            q_tau = np.quantile(y_ppred, tau, axis=0)
            QS += rho(y_true-q_tau, tau)
        return np.mean(QS)
