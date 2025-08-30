"""
Exponential Smoothing model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from benchmarking_pipeline.models.base_model import BaseModel


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Exponential Smoothing model with a given configuration.

        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12, ...}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config)

        # Get the model-specific config from the nested structure
        # BaseModel keeps the full config, so we need to extract the exponential_smoothing part

        def _cast_param(key, value):
            if key == "seasonal_periods":
                return int(value) if value is not None else None
            if key == "damped_trend":
                if isinstance(value, str):
                    return value.lower() == "true"
                return bool(value)
            if key == "forecast_horizon":
                return int(value) if value is not None else 1
            if key in ["trend", "seasonal"]:
                if isinstance(value, str) and value.lower() == "none":
                    return None
                return value
            return value

        if "trend" not in self.model_config:
            self.model_config["trend"] = None
        if "seasonal" not in self.model_config:
            self.model_config["seasonal"] = None
        if "seasonal_periods" not in self.model_config:
            self.model_config["seasonal_periods"] = None
        if "damped_trend" not in self.model_config:
            self.model_config["damped_trend"] = False

        self.model_config["trend"] = _cast_param("trend", self.model_config["trend"])
        self.model_config["seasonal"] = _cast_param(
            "seasonal", self.model_config["seasonal"]
        )
        self.model_config["seasonal_periods"] = _cast_param(
            "seasonal_periods", self.model_config["seasonal_periods"]
        )
        self.model_config["damped_trend"] = _cast_param(
            "damped_trend", self.model_config["damped_trend"]
        )

        # STRICT VALIDATION: Ensure parameters are meaningful
        if self.model_config["trend"] is None and self.model_config["seasonal"] is None:
            raise ValueError("At least one of 'trend' or 'seasonal' must be specified")

        if (
            self.model_config["seasonal"] is not None
            and self.model_config["seasonal_periods"] is None
        ):
            raise ValueError(
                "seasonal_periods must be specified when seasonal is specified"
            )
        if self.model_config["damped_trend"] and self.model_config["trend"] is None:
            raise ValueError("damped_trend can only be True when trend is specified")

        self.model_ = None

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "ExponentialSmoothingModel":

        print(
            f"[ExponentialSmoothing train] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}"
        )

        # Ensure correct types for model parameters
        trend = self.model_config["trend"]
        seasonal = self.model_config["seasonal"]

        if isinstance(trend, str) and trend.lower() == "none":
            trend = None

        if isinstance(seasonal, str) and seasonal.lower() == "none":
            seasonal = None

        seasonal_periods = (
            int(self.model_config["seasonal_periods"])
            if self.model_config["seasonal_periods"] is not None
            else None
        )

        damped_trend = bool(self.model_config["damped_trend"])
        if isinstance(damped_trend, str):
            damped_trend = damped_trend.lower() == "true"
        # Only allow damped_trend if trend is not None
        if trend is None:
            damped_trend = None

        # Handle input data - ensure we have the right format
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        elif isinstance(y_context, pd.DataFrame):
            endog = y_context.values.flatten()
        else:
            endog = y_context

        # Ensure endog is 1D
        if endog.ndim > 1:
            endog = endog.flatten()

        print(
            f"[ExponentialSmoothing train] endog shape: {endog.shape}, first 5 values: {endog[:5]}"
        )
        print(
            f"[ExponentialSmoothing train] parameters: trend={trend}, seasonal={seasonal}, seasonal_periods={seasonal_periods}, damped_trend={damped_trend}"
        )

        try:
            self.model_ = ExponentialSmoothing(
                endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend,
            ).fit()
            self.is_fitted = True
            print(f"[ExponentialSmoothing train] Model fitted successfully")
        except Exception as e:
            print(f"[ExponentialSmoothing train] Error fitting model: {e}")
            raise

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")

        forecast_steps = len(timestamps_target)
        print(f"[ExponentialSmoothing predict] Forecasting {forecast_steps} steps")

        try:
            forecast = self.model_.forecast(steps=forecast_steps)

            forecast = np.asarray(forecast)

            if len(forecast.shape) == 1:
                forecast = np.expand_dims(forecast, axis=1)

            print(f"[ExponentialSmoothing predict] result shape: {forecast.shape}")
            return forecast

        except Exception as e:
            print(f"[ExponentialSmoothing predict] Error during forecast: {e}")
            raise
