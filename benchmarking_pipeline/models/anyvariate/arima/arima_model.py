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
        self.full_config = config

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

    def _train(
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

    def _predict(
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

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ArimaModel":
        """
        Anyvariate wrapper: trains a separate ARIMA per variate if multivariate,
        or a single ARIMA in the univariate case.

        Assumes y_context and y_target are 2D ndarrays: (num_steps, num_features), even for univariate.
        """
        num_features = y_context.shape[1]
        # Multivariate: more than one feature (column)
        if num_features > 1:
            self.models = []
            for k in range(num_features):
                yc = y_context[:, k]    # Already 1D
                yt = y_target[:, k] if y_target is not None else None  # Already 1D
                # No need to reshape to 2D column; _train can handle 1D array for this variate
                m = ArimaModel(self.model_config)
                m._train(
                    y_context=yc,
                    y_target=yt,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                )
                self.models.append(m)
            # For compatibility, mirror first model state to top-level attributes
            self.model_ = self.models[0].model_
            self.is_fitted = True
            return self
        else:
            # Univariate: input is always (num_steps, 1)
            return self._train(
                y_context=y_context,
                y_target=y_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
            )

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Anyvariate wrapper: predicts per variate and stacks columns.
        """
        if hasattr(self, "models") and self.models:
            preds = []
            num_variates = len(self.models)
            for k, m in enumerate(self.models):
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(
                    y_context=yc,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                )
                preds.append(pk.reshape(-1, 1))
            return np.concatenate(preds, axis=1)
        # Univariate
        return self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            freq=freq,
        )
