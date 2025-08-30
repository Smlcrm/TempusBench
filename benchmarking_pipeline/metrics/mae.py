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
        """
        return np.mean(np.abs(y_true - y_pred))
