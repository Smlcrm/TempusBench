"""
Calculates Root Mean Squared Error.
"""

import numpy as np

from .base_metric import BaseMetric


class RMSE(BaseMetric):
    def __init__(self):
        super().__init__("deterministic")

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the RMSE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values (preprocessed by base class).
            **kwargs: Optional kwargs (unused)

        Returns:
            The calculated RMSE score.
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse.item()
