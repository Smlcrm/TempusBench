"""
Calculates Root Mean Squared Error.
"""
import numpy as np

from .base_metric import BaseMetric


class RMSE(BaseMetric):
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> float:
        """
        Computes the RMSE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Optional 'task_type' with value 'deterministic' or 'stochastic'.
                Defaults to 'deterministic' if not provided.

        Returns:
            The calculated RMSE score.
        """
        y_ppred = self.process_params(y_true, y_pred, **kwargs)
        return np.sqrt(np.mean((y_true - y_ppred) ** 2))