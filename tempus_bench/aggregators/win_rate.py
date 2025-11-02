"""
Calculates Average Win Rate.
"""

import pandas as pd
import numpy as np

from .base_aggregator import BaseAggregator


class WinRate(BaseAggregator):
    """
    Class to compute average win rate for models.

    Average win rate W_j represents the probability that model j achieves
    lower error than another randomly chosen model k != j on a randomly chosen task.
    """

    def __init__(self, pivot_table: pd.DataFrame):
        """
        Initialize the WinRate aggregator.

        Args:
            pivot_table: DataFrame with models as index, tasks as columns, scores as values
        """
        super().__init__(pivot_table)

    def __call__(self) -> pd.Series:
        """
        Compute average win rate for each model.

        Formula:
        W_j = (1 / (num_tasks * (num_models - 1))) * Σ_r Σ_k≠j [1(E_rj < E_rk) + 0.5 * 1(E_rj = E_rk)]

        Where:
        - num_tasks = number of tasks
        - num_models = number of models
        - E_rj = error of model j on task r
        - 1(condition) = indicator function (1 if true, 0 otherwise)

        Returns:
            Series with average win rate for each model
        """
        return self._compute()

    def _compute(self) -> pd.Series:
        """
        Compute average win rate for each model.

        Returns:
            Series with average win rate for each model
        """
        # Get numeric values only (handle NaN)
        data = self.pivot_table.values
        model_names = self.pivot_table.index

        num_tasks = data.shape[1]  # Number of tasks
        num_models = data.shape[0]  # Number of models

        win_rates = {}

        for j, model_j in enumerate(model_names):
            total_score = 0.0
            comparisons = 0

            for r in range(num_tasks):
                error_j = data[j, r]

                # Skip if model j has NaN for this task
                if pd.isna(error_j):
                    continue

                for k in range(num_models):
                    if k == j:  # Skip comparison with itself
                        continue

                    error_k = data[k, r]

                    # Skip if model k has NaN for this task
                    if pd.isna(error_k):
                        continue

                    comparisons += 1

                    if error_j < error_k:
                        # Model j wins (lower error is better)
                        total_score += 1.0
                    elif error_j == error_k:
                        # Tie - award 0.5 points
                        total_score += 0.5
                    # else: error_j > error_k, model j loses (0 points)

            # Average win rate
            if comparisons > 0:
                win_rates[model_j] = total_score / comparisons
            else:
                win_rates[model_j] = np.nan

        return pd.Series(win_rates, name="average_win_rate")
