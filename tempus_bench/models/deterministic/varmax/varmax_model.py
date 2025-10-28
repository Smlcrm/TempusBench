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
import warnings
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Literal
from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel

warnings.filterwarnings("ignore")


class VarmaxParams(PydanticBaseModel):
    p: int = Field(..., ge=0, description="Number of AR parameters")
    q: int = Field(..., ge=0, description="Number of MA parameters")
    trend: Literal["c", "t", "ct"] = Field(default="c", description="Deterministic trend: 'c' (constant), 't' (linear), 'ct' (both)")


class VARMAXModel(BaseModel):
    def __init__(self, config: JobConfig, logs_path: str):
        """
        Initialize VARMAX model with given configuration.

        Args:
            config: JobConfig instance containing model and task configuration
            logs_path: Directory for storing log files (required)
        """
        super().__init__(config, logs_path)
        
        # Validate and set model config using Pydantic
        self.model_config = VarmaxParams(**self.model_config).model_dump()

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