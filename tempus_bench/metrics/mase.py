import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Scaled Error.
"""


class MASE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Computes the MASE.
        Requires 'y_train' and optionally 'seasonal_period' in kwargs.
        Also accepts optional 'task_type' with value 'deterministic'.
        """

        task_type = kwargs.get('task_type', 'deterministic')

        if task_type != 'deterministic':
            raise ValueError(f"MASE can only be used with 'deterministic' task_type, got '{task_type}'.")

        y_train = kwargs.get("y_train")
        seasonal_period = kwargs.get("seasonal_period", 1)

        # Check if training data is provided and long enough for seasonal lag
        if y_train is None:
            return np.nan
        
        # For 2D arrays, check the time dimension (axis 0), for 1D arrays check the length
        if y_train.ndim == 2:
            if y_train.shape[0] <= seasonal_period:
                return np.nan
        else:
            if len(y_train) <= seasonal_period:
                return np.nan

        # Forecast errors
        forecast_errors = np.abs(y_true - y_pred)

        # Handle both 1D and 2D arrays
        if y_train.ndim == 1:
            # 1D case: single time series (num_steps,)
            # Calculate naive forecast errors: |y_t - y_{t-seasonal_period}|
            naive_errors = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
        else:
            # 2D case: multiple time series (num_steps, num_targets)
            # Calculate naive errors for each target and take the mean
            naive_errors_per_target = []
            for target_idx in range(y_train.shape[1]):
                target_series = y_train[:, target_idx]  # (num_steps,)
                # Calculate naive forecast errors: |y_t - y_{t-seasonal_period}|
                target_naive_errors = np.mean(np.abs(target_series[seasonal_period:] - target_series[:-seasonal_period]))
                naive_errors_per_target.append(target_naive_errors)
            naive_errors = np.mean(naive_errors_per_target)
        
        epsilon = 1e-10  # stability term

        return np.mean(forecast_errors / (naive_errors + epsilon))
