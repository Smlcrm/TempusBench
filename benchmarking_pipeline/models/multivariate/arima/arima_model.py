"""
Multivariate ARIMA model implementation.

This model extends the univariate ARIMA to handle multiple target variables simultaneously.
Uses Vector Autoregression (VAR) which is the multivariate extension of ARIMA models.
Design choices (similar to multivariate LSTM):
- Uses VAR from statsmodels for multivariate time series modeling
- Handles multiple targets simultaneously in one model
- Predicts all targets in a single forward pass
- Supports differencing for non-stationary series
"""

import os
import numpy as np
import math
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Union, Tuple, Optional
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


class MultivariateARIMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Multivariate ARIMA model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - p: int, VAR order (autoregressive order)
                - d: int, differencing order for stationarity
                - maxlags: int, maximum number of lags to consider
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)
        if "trend" not in self.model_config:
            raise ValueError("trend must be specified in config")
        if "maxlags" not in self.model_config:
            self.model_config["maxlags"] = 20
        if "ic" not in self.model_config:
            raise ValueError("ic must be specified in config")

        self.model = None

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "MultivariateARIMAModel":
        """
        Train the Multivariate ARIMA model on given data.

        TECHNIQUE: Vector Autoregression (VAR) for Multiple Time Series
        - Extends ARIMA to handle multiple interdependent time series
        - Each variable depends on its own lags and lags of other variables
        - Captures cross-dependencies between multiple targets
        - Applies differencing if needed to achieve stationarity
        - Uses Maximum Likelihood Estimation for parameter fitting

        Args:
            y_context: Past target values (time series) - used for training (can be DataFrame for multivariate)
            y_target: Future target values (optional, for extended training)
            y_start_date: The start date timestamp for y_context and y_target
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        if not self.is_fitted:
            self.model = VAR(
                endog=y_context, exog=None, dates=timestamps_context, freq=freq
            )

        self.results = self.model.fit(
            maxlags=self.model_config["maxlags"],
            ic=self.model_config["ic"],
            trend=self.model_config["trend"],
        )

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate ARIMA model.

        TECHNIQUE: VAR Forecasting for Multiple Time Series
        - Uses fitted VAR model to predict multiple steps ahead
        - Predicts all targets simultaneously using their interdependencies
        - Handles both in-sample and out-of-sample forecasting
        - Reverses differencing to get predictions in original scale

        Args:
            y_context: Past target values for prediction context
            y_target: Future target values (used to determine prediction length)
            x_context: Past exogenous variables (optional, ignored for now)
            x_target: Future exogenous variables (optional, ignored for now)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_horizon = timestamps_target.shape[0]
        lag_order = self.results.k_ar
        forecast_steps = len(timestamps_target)

        y_context = y_context[-lag_order:, :]
        forecasts = self.results.forecast(y_context, steps=forecast_steps)

        forecasts = np.array(forecasts)
        if len(forecasts.shape) == 1:
            forecasts = np.expand_dims(forecasts, axis=-1)

        return forecasts

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                # Convert from nanoseconds to datetime
                if timestamps[0] > 1e18:  # Likely nanoseconds
                    timestamps = pd.to_datetime(timestamps, unit="ns")
                elif timestamps[0] > 1e15:  # Likely microseconds
                    timestamps = pd.to_datetime(timestamps, unit="us")
                elif timestamps[0] > 1e12:  # Likely milliseconds
                    timestamps = pd.to_datetime(timestamps, unit="ms")
                else:  # Likely seconds
                    timestamps = pd.to_datetime(timestamps, unit="s")
            else:
                timestamps = pd.to_datetime(timestamps)
        else:
            timestamps = timestamps

        return timestamps
