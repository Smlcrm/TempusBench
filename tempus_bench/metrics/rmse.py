import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Root Mean Squared Error.
"""


class RMSE:
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

        task_type = kwargs.get('task_type')
        if task_type == 'stochastic':
            point_forecast_statistic = kwargs['point_forecast_statistic']
            if point_forecast_statistic == 'mean':
                y_ppred = np.mean(y_pred, axis=0)
            else:
                raise ValueError("MASE can only handle point_forecast_statistic == 'mean' for stochastic evaluation.")
        else:
            y_ppred = y_pred

        return np.sqrt(np.mean((y_true - y_ppred) ** 2))
