import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Quantile Score (QS) for deciles using the tilted ℓ₁ loss.
"""
class QuantileScore:
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
        task_type = kwargs.get('task_type')

        if task_type != 'stochastic':
            raise ValueError(f"QuantileScore can only be used with 'stochastic' task_type, got '{task_type}'.")

        S, T, M = y_pred.shape
        if y_true.shape != (T, M):
            raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but expected ({T}, {M}) to match y_pred (num_samples={S}, time_steps={T}, num_targets={M})")

        rho = lambda u, tau: np.maximum(tau * u, (tau - 1) * u)
        QS = np.zeros(y_true.shape)
        for tau in np.arange(0.0, 1.1, 0.1):
            q_tau = np.quantile(y_pred, tau, axis=0)
            QS += rho(y_true-q_tau, tau)
        return np.mean(QS)
