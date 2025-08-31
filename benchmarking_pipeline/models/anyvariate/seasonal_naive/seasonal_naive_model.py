"""
Seasonal Naive model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster
from benchmarking_pipeline.models.base_model import BaseModel


class SeasonalNaiveModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Seasonal Naive model with a given configuration.

        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'model_params': {'sp': 7}} for weekly seasonality in daily data.
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "SeasonalNaiveModel":
        """
        Train the Seasonal Naive model on given data. For this model, "training"
        simply means storing the historical data for future lookups.

        Args:
            y_context: Past target values (pd.Series or np.ndarray).
            y_target, x_context, x_target: Not used by this model, but included for compatibility.

        Returns:
            self: The fitted model instance.
        """

        if not self.is_fitted:
            if "sp" not in self.model_config:
                raise ValueError("sp must be specified in model_params")
            sp = self.model_config["sp"]
            self.model = NaiveForecaster(strategy="last", sp=sp)

        # Handle both 1D and 2D input data
        if isinstance(y_context, np.ndarray) and y_context.ndim == 2:
            # Extract the single column from 2D array
            y_context = y_context[:, 0]
        elif (
            hasattr(y_context, "values")
            and hasattr(y_context.values, "ndim")
            and y_context.values.ndim == 2
        ):
            # Handle pandas DataFrame or similar
            y_context = y_context.values[:, 0]

        if not isinstance(y_context, pd.Series):
            # works best with a proper index
            y_context = pd.Series(y_context)

        self.model.fit(y=y_context, X=None)
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
        Make predictions using the trained Seasonal Naive model.

        Args:
            y_context: Context time series values (pd.Series or np.ndarray).
            y_target: Used to determine the number of steps to forecast.
            x_context, x_target: Not used by this model, but included for compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon, 1).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        forecast_horizon = len(timestamps_target)
        fh = np.arange(1, forecast_horizon + 1)
        predictions = self.model.predict(fh=fh)

        if len(predictions.shape) == 1:
            predictions = np.asarray(predictions)
            predictions = np.expand_dims(predictions, axis=1)

        return predictions
