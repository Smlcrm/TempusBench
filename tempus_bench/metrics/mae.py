"""
Calculates Mean Absolute Error.
"""

import numpy as np

from .base_metric import BaseMetric


class MAE(BaseMetric):
    def __init__(self):
        super().__init__("deterministic")

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the MAE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values (preprocessed by base class).
            **kwargs: Optional kwargs (unused)

        Returns:
            The calculated MAE score.
        """
        return np.mean(np.abs(y_true - y_pred)).item()
