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

        num_samples = y_pred.shape[0]

        # CRPS formula: E|X - x| - E|X - X'|/2

        # Term 1: E|X - x| = mean(|samples - true_value|)
        # Broadcasting: (num_samples, ...) - (...) -> (num_samples, ...)
        term1 = np.mean(np.abs(y_pred - y_true), axis=0)

        # Term 2: E|X - X'|/2 = mean(|samples - samples')|/2
        # Compute pairwise differences between all samples
        # Reshape for broadcasting: (num_samples, 1, ...) - (1, num_samples, ...)
        y_pred_expanded1 = np.expand_dims(y_pred, axis=1)  # (num_samples, 1, ...)
        y_pred_expanded2 = np.expand_dims(y_pred, axis=0)  # (1, num_samples, ...)

        # Compute absolute differences: (num_samples, num_samples, ...)
        pairwise_diffs = np.abs(y_pred_expanded1 - y_pred_expanded2)

        # Take mean over sample pairs (first two dimensions), then over timesteps/targets
        term2 = np.mean(pairwise_diffs, axis=(0, 1)) / 2

        # CRPS = term1 - term2, then take mean across all dimensions
        crps = np.mean(term1 - term2)

        return float(crps)
