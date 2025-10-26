import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Error.
"""


class MAE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> float:
        """
        Computes the MAE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Optional 'task_type' with value 'deterministic' or 'stochastic'.
                Defaults to 'deterministic' if not provided.

        Returns:
            The calculated MAE score.
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
                raise ValueError("MAE can only handle point_forecast_statistic == 'mean' for stochastic evaluation.")

        else:
            y_ppred = y_pred

        return np.mean(np.abs(y_true - y_ppred))
