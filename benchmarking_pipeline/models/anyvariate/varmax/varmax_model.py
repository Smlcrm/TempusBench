"""
Multivariate VARMAX model.
"""

import os
import numpy as np
import math
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Union, Tuple, Optional
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


class VARMAXModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VARMAX model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - p: int, the number of AR parameters to use
                - d: int, the number of MA parameters to use
                - trend: string or iterable of ints, handles the deterministic trend
                         polynomail  (can be one of the following: 'c' - constant trend;
                         't' - linear trend with time; 'ct' - both 'c' and 't'; 
                         iterable of ints - represents the coefficients of each 
                         term of a polynomial that goes in increasing order. For example,
                         [1, 0, 1, 1] gives us a trend polynomial of a + ct^2 + dt^3).
                - training_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config)
        if "trend" not in self.model_config:
            raise ValueError("trend must be specified in config")
        if "p" not in self.model_config:
            raise ValueError("p must be specified in config")
        if "q" not in self.model_config:
            raise ValueError("q must be specified in config")

        self.model = None

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "VARMAXModel":
        """
        Train the Multivariate VARMAX model on given data.

        TECHNIQUE: Vector Autoregression (VAR) for Multiple Time Series
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
            self.model = VARMAX(
                endog=y_context, exog=None, 
                order=(self.model_config["p"], self.model_config["q"]), 
                trend=self.model_config["trend"]
            )

        self.results = self.model.fit()

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

        #forecast_horizon = timestamps_target.shape[0]
        #lag_order = self.results.k_ar
        #forecast_steps = len(timestamps_target)

        #y_context = y_context[-lag_order:, :]
        forecasts = self.results.forecast(steps=self.model_config["forecast_horizon"])

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
