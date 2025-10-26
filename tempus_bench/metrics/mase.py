import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Scaled Error.
"""

class MASE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> float:
        """
        Computes the MASE.
        Requires 'y_train' and optionally 'seasonal_period' in kwargs.
        Also accepts optional 'task_type' with value 'deterministic'.
        """

        task_type = kwargs.get('task_type')
        if task_type == 'stochastic':
            S, T, M = y_pred.shape
            if y_true.shape != (T, M):
                raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but expected ({T}, {M}) to match y_pred (num_samples={S}, time_steps={T}, num_targets={M})")

            point_forecast_statistic = kwargs['point_forecast_statistic']
            if point_forecast_statistic == 'mean':
                y_ppred = np.mean(y_pred, axis=0)
            else:
                raise ValueError("MASE can only handle point_forecast_statistic == 'mean' for stochastic evaluation.")
        else:
            y_ppred = y_pred

        denom = np.max(1e-10, np.mean(np.abs(y_true[1:]-y_true[:-1])))
        return np.mean(np.abs(y_true - y_ppred)) / denom
