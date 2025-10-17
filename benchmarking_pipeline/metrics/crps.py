import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Continuous Ranked Probability Score.
"""
class CRPS:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes the CRPS.
        Requires 'y_pred_dist_samples' in kwargs. y_pred is ignored.

        Args:
            y_true: True values, shape (n_timesteps,) or (n_timesteps, num_targets)
            y_pred: Point predictions (ignored, kept for API compatibility)
            **kwargs: Must contain 'y_pred_dist_samples' and optional 'task_type'
                - y_pred_dist_samples: shape (n_samples, n_timesteps) or (n_timesteps, num_targets, n_samples)
                - task_type: Optional, defaults to 'stochastic'

        Returns:
            CRPS score(s) as array with one score per timestep for univariate, or per target for multivariate
        """
        task_type = kwargs.get('task_type', 'stochastic')

        if task_type != 'stochastic':
            raise ValueError(f"CRPS can only be used with 'stochastic' task_type, got '{task_type}'.")

        y_pred_dist_samples = kwargs.get('y_pred_dist_samples')
        if y_pred_dist_samples is None:
            raise ValueError("y_pred_dist_samples must be provided in kwargs for CRPS.")

        y_s = np.asarray(y_pred_dist_samples)
        y_true = np.asarray(y_true)

        if y_true.ndim == 1:  # Univariate: (n_timesteps,)
            # y_s shape: (n_samples, n_timesteps)
            # y_true shape: (n_timesteps,)

            # Calculate CRPS for each timestep
            # y_s: (n_samples, n_timesteps), y_true: (n_timesteps,)
            # Broadcasting: y_true -> (1, n_timesteps)
            y_true_reshaped = y_true[np.newaxis, :]

            # Calculate CRPS for each timestep
            term1 = np.mean(np.abs(y_s - y_true_reshaped), axis=0)  # (n_timesteps,)

            # Calculate pairwise differences between samples
            n_samples = y_s.shape[0]
            if n_samples > 1:
                # Create all pairwise differences: (n_timesteps, n_samples, n_samples)
                # Transpose y_s to (n_timesteps, n_samples) for easier broadcasting
                y_s_t = y_s.T  # (n_timesteps, n_samples)
                diff_matrix = y_s_t[:, :, np.newaxis] - y_s_t[:, np.newaxis, :]
                term2 = np.mean(np.abs(diff_matrix), axis=(1, 2)) / 2  # (n_timesteps,)
            else:
                term2 = np.zeros_like(term1)

            # Return CRPS for each timestep (not the mean)
            return term1 - term2

        elif y_true.ndim == 2:  # Multivariate: (n_timesteps, num_targets)
            # y_s shape: (n_timesteps, num_targets, n_samples) from test
            # y_true shape: (n_timesteps, num_targets)

            num_targets = y_true.shape[1]
            crps_per_series = []

            for target_idx in range(num_targets):
                y_t_series = y_true[:, target_idx]  # (n_timesteps,)
                y_s_series = y_s[:, target_idx, :]  # (n_timesteps, n_samples)

                # Calculate CRPS for this target series
                y_t_reshaped = y_t_series[:, np.newaxis]  # (n_timesteps, 1)
                term1 = np.mean(np.abs(y_s_series - y_t_reshaped), axis=1)  # (n_timesteps,)

                n_samples = y_s_series.shape[1]
                if n_samples > 1:
                    # Create pairwise differences for this target
                    # y_s_series: (n_timesteps, n_samples)
                    diff_matrix = y_s_series[:, :, np.newaxis] - y_s_series[:, np.newaxis, :]
                    term2 = np.mean(np.abs(diff_matrix), axis=(1, 2)) / 2  # (n_timesteps,)
                else:
                    term2 = np.zeros_like(term1)

                # Mean CRPS for this target
                crps_per_series.append(np.mean(term1 - term2))

            return np.array(crps_per_series)

        else:
            raise ValueError(f"y_true must be 1D or 2D array, got {y_true.ndim}D")
