"""
Calculates Mean Absolute Error.
"""

import numpy as np

from .base_metric import BaseMetric


class MAE(BaseMetric):
    def __init__(self):
        super().__init__("deterministic")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the MAE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Optional kwargs (unused)

        Returns:
            The calculated MAE score.
        """
        return np.mean(np.abs(y_true - y_pred)).item()
