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
        """Train the VARMAX model. Requires multivariate (>=2 targets); univariate raises."""
        if y_context.ndim < 2 or y_context.shape[1] < 2:
            raise ValueError(
                "VARMAX requires at least 2 target variates; "
                f"got y_context.shape={y_context.shape}. Use ARIMA for univariate."
            )

        p = self.p
        q = self.q
        trend = self.trend

        self._has_exog = x_context is not None
        exog = None
        if x_context is not None:
            exog = np.asarray(x_context)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)

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
        """Forecast using the fitted VARMAX model."""
        if self._model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_steps = timestamps_target.shape[0]

        exog_oos = None
        if self._has_exog and x_target is not None:
            exog_oos = np.asarray(x_target[:forecast_steps])
            if exog_oos.ndim == 1:
                exog_oos = exog_oos.reshape(-1, 1)

        forecasts = self._model.forecast(steps=forecast_steps, exog=exog_oos)
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
