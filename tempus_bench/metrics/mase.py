import numpy as np

"""
Calculates Mean Absolute Scaled Error.
"""

from .base_metric import BaseMetric


class MASE(BaseMetric):
    def __init__(self):
        super().__init__("deterministic")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes the MASE.
        Requires 'y_train' and optionally 'seasonal_period' in kwargs.
        """
        denom = np.maximum(1e-10, np.mean(np.abs(y_true[1:] - y_true[:-1])))
        mase = np.mean(np.abs(y_true - y_pred)) / denom
        return mase.item()
