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
from tempus_bench.models.base_model import BaseModel
from tempus_bench.utils.logger import get_logger

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

        # Initialize logger if logs_dir is available
        logging_config = config.get('logging')
        logs_dir = logging_config.get('logs_dir')
        self.logger = get_logger(logs_dir)

        # Extract ARIMA-specific parameters
        if "p" not in self.model_config:
            raise ValueError("p must be specified in config")
        if "d" not in self.model_config:
            raise ValueError("d must be specified in config")
        if "q" not in self.model_config:
            raise ValueError("q must be specified in config")
        if "s" not in self.model_config:
            raise ValueError("s must be specified in config")

        # Log initialization parameters
        if self.logger:
            self.logger.debug("ARIMA Init", f"Initializing ARIMA model with parameters: p={self.model_config['p']}, d={self.model_config['d']}, q={self.model_config['q']}, s={self.model_config['s']}")
            self.logger.debug("ARIMA Init", f"Full config received: {self.full_config}")

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
        if self.logger:
            self.logger.debug("ARIMA Train", f"Starting training with input shapes: y_context={y_context.shape}, y_target={y_target.shape if y_target is not None else None}")
            self.logger.debug("ARIMA Train", f"Timestamps context range: {timestamps_context[0] if len(timestamps_context) > 0 else 'empty'} to {timestamps_context[-1] if len(timestamps_context) > 0 else 'empty'}")
            self.logger.debug("ARIMA Train", f"Timestamps target range: {timestamps_target[0] if len(timestamps_target) > 0 else 'empty'} to {timestamps_target[-1] if len(timestamps_target) > 0 else 'empty'}")
            self.logger.debug("ARIMA Train", f"Frequency parameter: {freq}")

        # Ensure endogenous series is 1D for statsmodels
        endog = y_context.squeeze()
        
        if self.logger:
            self.logger.debug("ARIMA Train", f"Endogenous series shape after squeeze: {endog.shape}")
            self.logger.debug("ARIMA Train", f"Endogenous series sample values: {endog[:5] if len(endog) >= 5 else endog}")

        # No exogenous variables supported
        exog = None
        if self.logger:
            self.logger.debug("ARIMA Train", f"Exogenous variables: {exog} (None - not supported)")

        # Use seasonal_order only if seasonal period is greater than 1
        if self.model_config["s"] > 1:
            if self.logger:
                self.logger.debug("ARIMA Train", f"Using seasonal ARIMA with seasonal period s={self.model_config['s']}")
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
            if self.logger:
                self.logger.debug("ARIMA Train", f"Using non-seasonal ARIMA")
            model = ARIMA(
                endog=endog,
                order=(
                    self.model_config["p"],
                    self.model_config["d"],
                    self.model_config["q"],
                ),
                exog=exog,
            )

        if self.logger:
            self.logger.debug("ARIMA Train", f"Model order parameters: p={self.model_config['p']}, d={self.model_config['d']}, q={self.model_config['q']}, s={self.model_config['s']}")
            self.logger.debug("ARIMA Train", f"Starting model fitting...")

        self.model_ = model.fit()
        
        if self.logger:
            self.logger.debug("ARIMA Train", f"Model fitted successfully")
            self.logger.debug("ARIMA Train", f"Model summary: {self.model_.summary()}")
            self.logger.debug("ARIMA Train", f"Model AIC: {self.model_.aic}, BIC: {self.model_.bic}")
            self.logger.debug("ARIMA Train", f"Model coefficients: {self.model_.params}")
            self.logger.info("ARIMA Train", "Training completed successfully")
        
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
        if self.logger:
            self.logger.debug("ARIMA Predict", f"Starting prediction with context shape: {y_context.shape}")
            self.logger.debug("ARIMA Predict", f"Timestamps context: {timestamps_context[0] if len(timestamps_context) > 0 else 'empty'} to {timestamps_context[-1] if len(timestamps_context) > 0 else 'empty'}")
            self.logger.debug("ARIMA Predict", f"Timestamps target: {timestamps_target[0] if len(timestamps_target) > 0 else 'empty'} to {timestamps_target[-1] if len(timestamps_target) > 0 else 'empty'}")

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

        if self.logger:
            self.logger.debug("ARIMA Predict", f"Frequency validation passed: {freq}")
            self.logger.debug("ARIMA Predict", f"Forecast steps calculated from timestamps_target: {forecast_steps}")
            self.logger.debug("ARIMA Predict", f"Generating forecast...")

        forecast = self.model_.forecast(steps=forecast_steps, exog=None)
        forecast_array = np.asarray(forecast)

        if self.logger:
            self.logger.debug("ARIMA Predict", f"Forecast values generated: {forecast_array}")
            self.logger.debug("ARIMA Predict", f"Forecast array shape: {forecast_array.shape}")

        self._last_y_pred = forecast_array.reshape(-1, 1)

        if self.logger:
            self.logger.debug("ARIMA Predict", f"Final prediction shape: {self._last_y_pred.shape}")
            self.logger.info("ARIMA Predict", "Prediction completed successfully")

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
        
        if self.logger:
            self.logger.debug("ARIMA Train Wrapper", f"Number of features/variates detected: {num_features}")
        
        # Multivariate: more than one feature (column)
        if num_features > 1:
            if self.logger:
                self.logger.debug("ARIMA Train Wrapper", "Taking multivariate path - training separate ARIMA per variate")
            self.models = []
            for k in range(num_features):
                if self.logger:
                    self.logger.debug("ARIMA Train Wrapper", f"Training variate k={k}")
                yc = y_context[:, k]    # Already 1D
                yt = y_target[:, k] if y_target is not None else None  # Already 1D
                # No need to reshape to 2D column; _train can handle 1D array for this variate
                m = ArimaModel(self.config)
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
            if self.logger:
                self.logger.info("ARIMA Train Wrapper", f"Multivariate training completed for {num_features} variates")
            return self
        else:
            # Univariate: input is always (num_steps, 1)
            if self.logger:
                self.logger.debug("ARIMA Train Wrapper", "Taking univariate path")
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
            if self.logger:
                self.logger.debug("ARIMA Predict Wrapper", f"Multivariate prediction for {len(self.models)} variates")
            preds = []
            num_variates = len(self.models)
            for k, m in enumerate(self.models):
                if self.logger:
                    self.logger.debug("ARIMA Predict Wrapper", f"Predicting for variate k={k}")
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(
                    y_context=yc,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                )
                preds.append(pk.reshape(-1, 1))
            result = np.concatenate(preds, axis=1)
            if self.logger:
                self.logger.debug("ARIMA Predict Wrapper", f"Final concatenated prediction shape: {result.shape}")
                self.logger.info("ARIMA Predict Wrapper", "Multivariate prediction completed successfully")
            return result
        # Univariate
        if self.logger:
            self.logger.debug("ARIMA Predict Wrapper", "Univariate prediction")
        return self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            freq=freq,
        )
