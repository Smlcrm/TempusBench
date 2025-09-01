"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides an ARIMA model implementation for univariate time series forecasting.
ARIMA models combine autoregression, differencing, and moving average components to
capture temporal dependencies in time series data.

The model supports both seasonal and non-seasonal ARIMA variants and can handle
exogenous variables for enhanced forecasting performance.
"""

import pdb
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel


class ArimaModel(BaseModel):
    """
    ARIMA model for univariate time series forecasting.

    This class implements the ARIMA model with support for:
    - Non-seasonal ARIMA(p,d,q) models
    - Seasonal ARIMA(p,d,q)(P,D,Q,s) models
    - Exogenous variable support
    - Rolling window predictions
    - Model persistence and loading

    Attributes:
        p: AR order (autoregressive)
        d: Differencing order (integration)
        q: MA order (moving average)
        s: Seasonality period
        model_: Fitted ARIMA model instance
        loss_function: Loss function for training
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ARIMA model with given configuration.

        Args:
            config: Configuration dictionary containing model parameters
                - p: int, AR order (default: 1)
                - d: int, differencing order (default: 1)
                - q: int, MA order (default: 1)
                - s: int, seasonality period (default: 1)
                - loss_function: str, loss function for training (default: 'mae')
                - forecast_horizon: int, number of steps to forecast ahead
        """
        super().__init__(config)

        # Extract ARIMA-specific parameters
        if "p" not in self.model_config:
            raise ValueError("p must be specified in config")
        if "d" not in self.model_config:
            raise ValueError("d must be specified in config")
        if "q" not in self.model_config:
            raise ValueError("q must be specified in config")
        if "s" not in self.model_config:
            raise ValueError("s must be specified in config")

        # Initialize model state
        self.model_ = None
        self.is_fitted = False

        # forecast_horizon is inherited from parent class (BaseModel)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ArimaModel":
        """
        Train the ARIMA model on given data.

        Args:
            y_context: Past target values - training data (required)
            y_target: Future target values (not used in training, for compatibility)
            timestamps_context: Timestamps for y_context (not used in ARIMA)
            timestamps_target: Timestamps for y_target (not used in ARIMA)
            freq: Frequency string (required by interface, not used in ARIMA)

        Returns:
            self: The fitted model instance

        Note:
            ARIMA models only use y_context for training.
            y_target, timestamps_context, timestamps_target, and freq are ignored to prevent data leakage.
        """
        # Convert y_context to numpy array if needed
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context

        # No exogenous variables supported
        exog = None

        # Use seasonal_order only if seasonal period is greater than 1
        if self.model_config["s"] > 1:
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config["p"],
                    self.model_config["d"],
                    self.model_config["q"],
                ),
                seasonal_order=(0, 0, 0, self.model_config["s"]),
                exog=exog,
            )
        else:
            # Non-seasonal ARIMA
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config["p"],
                    self.model_config["d"],
                    self.model_config["q"],
                ),
                exog=exog,
            )

        self.model_ = model.fit()
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model, rolling forward using the fitted model.

        Args:
            y_context: Recent/past target values (not used for ARIMA prediction)
            timestamps_context: Timestamps for y_context (not used for ARIMA prediction)
            timestamps_target: Timestamps for the prediction horizon (used to determine forecast length)
            freq: Frequency string (must be provided from CSV data, required)

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If model is not fitted, freq is not provided, or forecast length cannot be determined
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        if freq is None or freq == "":
            raise ValueError(
                "Frequency (freq) must be provided from CSV data. Cannot use defaults or fallbacks."
            )

        if timestamps_target is None:
            raise ValueError(
                "timestamps_target must be provided to determine forecast horizon for ARIMA."
            )
        forecast_steps = len(timestamps_target)
        if forecast_steps <= 0:
            raise ValueError(
                "Forecast horizon must be positive (timestamps_target must be non-empty)."
            )

        forecast = self.model_.forecast(steps=forecast_steps, exog=None)
        forecast_array = np.asarray(forecast)

        self._last_y_pred = forecast_array.reshape(-1, 1)

        return self._last_y_pred
