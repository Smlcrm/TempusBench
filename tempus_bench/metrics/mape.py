import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Percentage Error.
"""


class MAPE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> float:
        """
        Computes the MAPE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: 'task_type' and 'point_forecast_statistic'

        Returns:
            The calculated MAPE score in percentage.
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
                raise ValueError("MAPE can only handle point_forecast_statistic == 'mean' for stochastic evaluation.")
        else:
            y_ppred = y_pred
        epsilon = 1e-10  # stability term to avoid division by zero
        denom = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs((y_true - y_ppred) / denom)) * 100
