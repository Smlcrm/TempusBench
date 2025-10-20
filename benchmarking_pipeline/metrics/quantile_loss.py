import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Quantile Loss (Pinball Loss).
"""
class QuantileLoss:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes Quantile Loss.
        Requires 'y_pred_quantiles' and 'quantiles_q_values' in kwargs. y_pred is ignored.
        Also accepts optional 'task_type' with value 'stochastic'.
        """
        task_type = kwargs.get('task_type', 'stochastic')

        if task_type != 'stochastic':
            raise ValueError(f"QuantileLoss can only be used with 'stochastic' task_type, got '{task_type}'.")

        y_pred_quantiles = kwargs.get('y_pred_quantiles')
        q_values = kwargs.get('quantiles_q_values')
        if y_pred_quantiles is None or q_values is None:
            raise ValueError("y_pred_quantiles and quantiles_q_values must be in kwargs for QuantileLoss.")

        losses = []
        for i, q in enumerate(q_values):
            errors = y_true - y_pred_quantiles[..., i]
            loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
            losses.append(loss)
        return np.mean(losses)