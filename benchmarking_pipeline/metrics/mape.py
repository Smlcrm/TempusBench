import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Percentage Error.
"""


class MAPE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Computes the MAPE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Ignored, for API consistency.

        Returns:
            The calculated MAPE score in percentage.
        """
        epsilon = 1e-10  # stability term to avoid division by zero
        denom = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
