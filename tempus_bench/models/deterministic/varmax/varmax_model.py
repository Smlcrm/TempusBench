"""
Multivariate VARMAX model.
"""
import math
import os
import pickle
import warnings

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller

from ...base_model import BaseModel, validate_inputs

warnings.filterwarnings("ignore")


class VarmaxHyperparams(PydanticBaseModel):
    p: int = Field(..., ge=0, description="Number of AR parameters")
    q: int = Field(..., ge=0, description="Number of MA parameters")
    trend: Optional[Literal["c", "t", "ct"]] = Field(default="c", description="Deterministic trend: 'c' (constant), 't' (linear), 'ct' (both)")


class VARMAXModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize VARMAX model with model-specific parameters.
        """
        super().__init__(params, settings, VarmaxHyperparams)

        self._model = None

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
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
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        p = self.p
        q = self.q
        trend = self.trend
        
        timestamps_context = self.convert_to_datetimeindex(timestamps_context)
        if not self.is_fitted:
            self._model = VARMAX(
                endog=y_context, exog=None, 
                order=(p, q), 
                trend=trend
            )

        self.results = self._model.fit()

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
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
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, num_targets)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        p = self.p
        q = self.q
        trend = self.trend
        
        if self._model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_steps = len(timestamps_target)
        forecasts = self.results.forecast(steps=forecast_steps)

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