import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Error.
"""


class MAE:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Computes the MAE.

        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Optional 'task_type' with value 'deterministic' or 'stochastic'.
                Defaults to 'deterministic' if not provided.

        Returns:
            The calculated MAE score.
        """
        task_type = kwargs.get('task_type', 'deterministic')

        if task_type not in ['deterministic', 'stochastic']:
            raise ValueError(f"Invalid task_type '{task_type}'. Must be 'deterministic' or 'stochastic'.")

        return np.mean(np.abs(y_true - y_pred))
