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
        """
        y_train = kwargs.get("y_train")
        seasonal_period = kwargs.get("seasonal_period", 1)

        # Check if training data is long enough for seasonal lag
        if len(y_train) <= seasonal_period:
            return np.nan

        # Forecast errors
        forecast_errors = np.abs(y_true - y_pred)

        # Naive seasonal errors using np.diff
        naive_errors = np.mean(
            np.abs(np.diff(y_train, n=seasonal_period, axis=0)), axis=0, keepdims=True
        )
        epsilon = 1e-10  # stability term

        return np.mean(forecast_errors / (naive_errors + epsilon))
