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

from tempus_bench.models.base_model import BaseModel, validate_inputs

warnings.filterwarnings("ignore")


class VarmaxHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    p: int = Field(..., ge=0, description="Number of AR parameters")
    q: int = Field(..., ge=0, description="Number of MA parameters")
    # Fixed Hyperparameters - Optional for User to override
    trend: Literal["c", "t", "ct"] = Field(
        default="c",
        description="Deterministic trend: 'c' (constant), 't' (linear), 'ct' (both)",
    )


class VarmaxModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, VarmaxHyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "VarmaxModel":
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
        p = self.p
        q = self.q
        trend = self.trend

        exog = None
        if x_context is not None:
            exog = x_context.squeeze()

        timestamps_context = self._convert_to_datetimeindex(timestamps_context)
        if not self.is_fitted:
            model = VARMAX(endog=y_context, exog=exog, order=(p, q), trend=trend)

            self._model = model.fit()

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
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

        if self._model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_steps = timestamps_target.shape[0]
        forecasts = self._model.forecast(steps=forecast_steps)
        return np.asarray(forecasts)  # (forecast_steps, num_targets)

    def _convert_to_datetimeindex(self, timestamps):
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
