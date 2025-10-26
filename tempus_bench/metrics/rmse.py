import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Root Mean Squared Error.
"""


class RMSE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Union[float, np.ndarray]:
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
        task_type = kwargs.get('task_type', 'deterministic')

        if task_type not in ['deterministic', 'stochastic']:
            raise ValueError(f"Invalid task_type '{task_type}'. Must be 'deterministic' or 'stochastic'.")

        return np.sqrt(np.mean((y_true - y_pred) ** 2))
