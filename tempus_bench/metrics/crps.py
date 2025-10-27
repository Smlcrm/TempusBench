import numpy as np
from typing import Dict, Any, Union

"""
Calculates Continuous Ranked Probability Score.
"""
class CRPS:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the CRPS using fully vectorized operations.

        Args:
            y_true: True values, shape (n_timesteps,) or (n_timesteps, num_targets)
            y_pred: Sample predictions, shape (num_samples, n_timesteps) or (num_samples, n_timesteps, num_targets)
            **kwargs: Optional parameters
                - task_type: Optional, defaults to 'stochastic'

        Returns:
            CRPS score as float (mean across all timesteps and targets)
        """
        task_type = kwargs.get('task_type')
        if task_type != 'stochastic':
            raise ValueError(f"CRPS can only be used with 'stochastic' task_type, got '{task_type}'.")
        S, T, M = y_pred.shape
        if y_true.shape != (T, M):
            raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but expected ({T}, {M}) to match y_pred (num_samples={S}, time_steps={T}, num_targets={M})")

        # Assuming i.i.d. samples, y_pred_1, ..., y_pred_N
        # We estimate the unbiased Monte-Carlo Estimator for CRPS

        crps = np.mean(np.abs(y_pred - y_true[None, ...]), axis=0)
        y_pred_sort = np.sort(y_pred, axis=0)
        j = np.arange(1, S + 1)[:, None, None]
        coeff = (2 * j - S - 1)
        crps -= np.sum(coeff * y_pred_sort, axis=0) / (S**2)

        return float(np.mean(crps))
