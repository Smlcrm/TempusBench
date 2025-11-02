"""
Calculates Skill Score.
"""

import pandas as pd
import numpy as np

from .base_aggregator import BaseAggregator


class SkillScore(BaseAggregator):
    """
    Class to compute skill score for models compared to a baseline.
    
    Skill score S_j quantifies how much model j reduces forecasting error
    compared to the fixed baseline model β on average.
    """
    
    def __init__(
        self, pivot_table: pd.DataFrame, baseline_model: str = "seasonal_naive"
    ):
        """
        Initialize the SkillScore aggregator.
        
        Args:
            pivot_table: DataFrame with models as index, tasks as columns, scores as values
            baseline_model: Name of the baseline model for skill score calculation
        """
        super().__init__(pivot_table)
        self.baseline_model = baseline_model
        
        # Validate that baseline model exists
        if baseline_model not in self.pivot_table.index:
            raise ValueError(
                f"Baseline model '{baseline_model}' not found in pivot table index. "
                f"Available models: {list(self.pivot_table.index)}"
            )
    
    def __call__(self) -> pd.Series:
        """
        Compute skill score for each model compared to baseline.
        
        Formula:
        S_j = 1 - (Π_r clip(E_rj / E_rβ; l, u))^(1/R)
        
        Where:
        - E_rj = error of model j on task r
        - E_rβ = error of baseline model β on task r
        - clip(x; l, u) = max(l, min(x, u))
        - l = 10^-2 = 0.01 (lower bound)
        - u = 100 (upper bound)
        - R = number of tasks
        
        Returns:
            Series with skill score for each model
        """
        return self._compute()
    
    def _compute(self) -> pd.Series:
        """
        Compute skill score for each model compared to baseline.
        
        Returns:
            Series with skill score for each model
        """
        # Get baseline model errors
        baseline_idx = self.pivot_table.index.get_loc(self.baseline_model)
        baseline_errors = self.pivot_table.iloc[baseline_idx].values

        # Get all model errors
        data = self.pivot_table.values
        model_names = self.pivot_table.index

        skill_scores = {}

        for j, model_j in enumerate(model_names):
            relative_errors = []

            for r in range(data.shape[1]):
                error_j = data[j, r]
                error_baseline = baseline_errors[r]

                # Skip if either value is NaN
                if pd.isna(error_j) or pd.isna(error_baseline):
                    continue

                # Avoid division by zero
                if error_baseline == 0:
                    if error_j == 0:
                        # Both zero - treat as 1.0 (equal performance)
                        relative_error = 1.0
                    else:
                        # Baseline is zero but model has error - treat as worst case
                        relative_error = 100.0
                else:
                    relative_error = error_j / error_baseline

                # Clip the relative error
                clipped_error = self._clip(relative_error, lower=1e-2, upper=100.0)
                relative_errors.append(clipped_error)

            # Compute geometric mean
            if len(relative_errors) > 0:
                # Geometric mean = (product)^(1/n)
                geometric_mean = np.prod(relative_errors) ** (
                    1.0 / len(relative_errors)
                )
                skill_score = 1.0 - geometric_mean
                skill_scores[model_j] = skill_score
            else:
                skill_scores[model_j] = np.nan

        return pd.Series(skill_scores, name="skill_score")

