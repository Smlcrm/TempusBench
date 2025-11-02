"""Aggregation metrics for model performance evaluation."""

import pandas as pd
import numpy as np
from typing import Optional


class BaseAggregator:
    """
    Base class to aggregate model scores across tasks.

    Takes a pandas DataFrame with:
    - Rows: model names
    - Columns: task names
    - Values: scores (e.g., error metrics where lower is better)

    Subclasses should implement the __call__ method to compute specific aggregation scores.
    """

    def __init__(self, pivot_table: pd.DataFrame):
        """
        Initialize the BaseAggregator.

        Args:
            pivot_table: DataFrame with models as index, tasks as columns, scores as values
        """
        self.pivot_table = pivot_table.copy()

    def _clip(
        self, x: np.ndarray, lower: float = 1e-2, upper: float = 100.0
    ) -> np.ndarray:
        """
        Clip values to be within [lower, upper].

        Args:
            x: Array of values to clip
            lower: Lower bound (default: 0.01)
            upper: Upper bound (default: 100.0)

        Returns:
            Clipped array
        """
        return np.clip(x, lower, upper)
