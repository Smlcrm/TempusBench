"""
Calculates Mean Absolute Percentage Error.
"""

import numpy as np

from .base_metric import BaseMetric


class MAPE(BaseMetric):
    def __init__(self):
        super().__init__("deterministic")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the MAPE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Optional kwargs (unused)

        Returns:
            The calculated MAPE score in percentage.
        """
        epsilon = 1e-10  # stability term to avoid division by zero
        denom = np.maximum(np.abs(y_true), epsilon)
        mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
        return mape.item()
