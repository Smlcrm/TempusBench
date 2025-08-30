"""
Croston's Classic Model implementation for intermittent demand forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
import pickle
import os


from benchmarking_pipeline.models.base_model import BaseModel


class CrostonClassicModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Croston's Classic model with a given configuration.

        Args:
            config: Configuration dictionary containing model parameters.
                - alpha: float, smoothing parameter for demand level (0 < alpha < 1)
                - gamma: float, smoothing parameter for interval level (0 < gamma < 1)
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)

        # Parameters, initialized to None
        self.demand_level_ = None
        self.interval_level_ = None

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray = None,
        x_context: np.ndarray = None,
        x_target: np.ndarray = None,
        **kwargs,
    ) -> "CrostonClassicModel":
        """
        Train the Croston's Classic model on the given time series data.

        The method decomposes the series into non-zero demand values and the
        time intervals between them, then applies Simple Exponential Smoothing
        to both series.

        Args:
            y_context: Past target values - the training data for the time series.
            y_target: Not used by the Croston's Classic model, but included for compatibility with the base class.

        Returns:
            self: The fitted model instance.
        """

        alpha = self.model_config["alpha"]
        gamma = self.model_config["gamma"]
        num_variates = y_context.shape[1]

        demand_levels = np.zeros(num_variates)
        interval_levels = np.zeros(num_variates)

        for variate in range(num_variates):

            serie = y_context[:, variate]
            # Find indices of non-zero demand
            non_zero_indices = np.nonzero(serie)

            # If there's no demand or only one demand point we cannot forecast
            if len(non_zero_indices[0]) < 2:

                # Set levels to a default state (e.g., average of the whole series)
                demand_levels[variate] = np.mean(serie) if len(serie) > 0 else 0
                interval_levels[variate] = len(series) if len(serie) > 0 else 1

            else:
                self.is_fitted = True

                # Get demand values and intervals
                demands = serie[non_zero_indices]
                intervals = np.diff(non_zero_indices[0])

                # Demand level starts with the first non-zero demand
                current_demand_level = demands[0]

                # Interval level starts with the first interval
                current_interval_level = intervals[0] if len(intervals) > 0 else 1

                # Apply SES to the rest of the demands and intervals
                for i in range(1, len(demands)):
                    current_demand_level = (
                        alpha * demands[i]
                        + (1 - alpha) * current_demand_level
                    )

                for i in range(1, len(intervals)):
                    current_interval_level = (
                        gamma * intervals[i]
                        + (1 - gamma) * current_interval_level
                    )

                demand_levels[variate] = current_demand_level
                interval_levels[variate] = current_interval_level

        self.demand_level_ = demand_levels
        self.interval_level_ = interval_levels
        self.is_fitted = True

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):
        """
        Make predictions using the trained Croston's Classic model.

        Args:
            y_target: Used to determine the number of steps to forecast.
            y_context, x_context, x_target: Not used.

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon, num_target_features).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        forecast_horizon = len(timestamps_target)
        forecast = np.divide(
            self.demand_level_,
            self.interval_level_,
            out=np.zeros_like(self.demand_level_, dtype=float),
            where=self.interval_level_ != 0,
        )
        forecast = np.tile(forecast, reps=(forecast_horizon, 1))

        return forecast.reshape(1, -1)  # Reshape to (1, forecast_steps)


    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Croston's Classic model's properties.

        Returns:
            Dict[str, Any]: A dictionary containing model summary information.
        """
        summary = {
            "model_type": "CrostonClassic",
            "alpha": self.alpha,
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            summary.update(
                {
                    "fitted_demand_level": self.demand_level_,
                    "fitted_interval_level": self.interval_level_,
                }
            )

        return summary
