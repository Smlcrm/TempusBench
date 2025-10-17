import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates the Interval Score (Winkler Score).
"""
class IntervalScore:
    def __init__(self, alpha: float = 0.1):
        """
        Initialize IntervalScore with alpha parameter.

        Args:
            alpha: Significance level for the prediction interval (default: 0.1)
        """
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes Interval Score.

        Args:
            y_true: True values
            y_pred: Point predictions (ignored, kept for API compatibility) or lower bounds
            *args: Additional arguments - if provided, args[0] is upper bounds
            **kwargs: Can contain:
                - y_pred_lower_bound: Lower bounds of prediction intervals
                - y_pred_upper_bound: Upper bounds of prediction intervals
                - interval_alpha: Override alpha from constructor
                - task_type: Optional, defaults to 'stochastic'

        Returns:
            Interval score(s) as array with one score per timestep
        """
        task_type = kwargs.get('task_type', 'stochastic')

        if task_type != 'stochastic':
            raise ValueError(f"IntervalScore can only be used with 'stochastic' task_type, got '{task_type}'.")
        # Check if bounds are provided as separate arguments
        if len(args) > 0:
            # y_pred is actually lower bounds, args[0] is upper bounds
            lower = y_pred
            upper = args[0]
        elif len(kwargs) == 0:
            # Try to get from y_pred if it's a tuple/list with (lower, upper)
            if isinstance(y_pred, (tuple, list)) and len(y_pred) == 2:
                lower, upper = y_pred
            else:
                raise ValueError("Must provide lower and upper bounds either as separate arguments, kwargs, or as y_pred=(lower, upper)")
        else:
            # Get from kwargs
            lower = kwargs.get('y_pred_lower_bound')
            upper = kwargs.get('y_pred_upper_bound')

        # Allow alpha override from kwargs
        alpha = kwargs.get('interval_alpha', self.alpha)

        if lower is None or upper is None:
            raise ValueError("Lower and upper bounds must be provided for IntervalScore.")

        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Validate shapes
        if y_true.shape != lower.shape or y_true.shape != upper.shape:
            raise ValueError("y_true, lower, and upper must have the same shape")

        # Calculate interval score for each timestep
        width = upper - lower
        penalty_lower = (2 / alpha) * np.maximum(0, lower - y_true)
        penalty_upper = (2 / alpha) * np.maximum(0, y_true - upper)
        score = width + penalty_lower + penalty_upper

        # Return scores for each timestep (not the mean)
        return score