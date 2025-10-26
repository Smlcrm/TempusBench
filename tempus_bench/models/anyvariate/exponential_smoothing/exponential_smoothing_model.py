"""
Exponential Smoothing model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tempus_bench.models.base_model import BaseModel
from tempus_bench.utils.logger import get_logger


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
        # Get logs directory from config, default to 'logs' if not specified
        logs_dir = config['logging']['logs_dir']
        self.logger = get_logger(logs_dir)

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

        # Cast parameters to correct types (no defaults - all must be in config)
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

        self.model = None

    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "ExponentialSmoothingModel":
        """
        Train a single Exponential Smoothing model on univariate data.
        """
        self.logger.debug("ExponentialSmoothingModel", f"y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")

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

        # Ensure endog is 1D for univariate case
        if endog.ndim > 1:
            endog = endog.squeeze()

        self.logger.debug("ExponentialSmoothingModel", f"endog shape: {endog.shape}, first 5 values: {endog[:5]}")
        self.logger.debug("ExponentialSmoothingModel", f"parameters: trend={trend}, seasonal={seasonal}, seasonal_periods={seasonal_periods}, damped_trend={damped_trend}")

        try:
            self.model = ExponentialSmoothing(
                endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend,
            ).fit()
            self.is_fitted = True
            self.logger.info("ExponentialSmoothingModel", "Model fitted successfully")
        except Exception as e:
            self.logger.error("ExponentialSmoothingModel", f"Error fitting model: {e}")
            raise

        return self

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "ExponentialSmoothingModel":
        """
        Anyvariate wrapper: trains a separate Exponential Smoothing per variate if multivariate,
        or a single Exponential Smoothing in the univariate case.

        Assumes y_context and y_target are 2D ndarrays: (num_steps, num_features), even for univariate.
        """
        num_features = y_context.shape[1]
        
        if self.logger:
            self.logger.debug("ExponentialSmoothing Train Wrapper", f"Number of features/variates detected: {num_features}")
        
        # Multivariate: more than one feature (column)
        if num_features > 1:
            if self.logger:
                self.logger.debug("ExponentialSmoothing Train Wrapper", "Taking multivariate path - training separate Exponential Smoothing per variate")
            self.models = []
            for k in range(num_features):
                if self.logger:
                    self.logger.debug("ExponentialSmoothing Train Wrapper", f"Training variate k={k}")
                yc = y_context[:, k]    # Already 1D
                yt = y_target[:, k] if y_target is not None else None  # Already 1D
                # Create new model instance for this variate
                m = ExponentialSmoothingModel(self.config)
                m._train(
                    y_context=yc,
                    y_target=yt,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                    **kwargs,
                )
                self.models.append(m)
            # For compatibility, mirror first model state to top-level attributes
            self.model = self.models[0].model
            self.is_fitted = True
            if self.logger:
                self.logger.info("ExponentialSmoothing Train Wrapper", f"Multivariate training completed for {num_features} variates")
            return self
        else:
            # Univariate: input is always (num_steps, 1)
            if self.logger:
                self.logger.debug("ExponentialSmoothing Train Wrapper", "Taking univariate path")
            return self._train(
                y_context=y_context,
                y_target=y_target,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                **kwargs,
            )

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict using a single Exponential Smoothing model on univariate data.
        """
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")

        forecast_steps = len(timestamps_target)
        self.logger.debug("ExponentialSmoothingModel", f"Forecasting {forecast_steps} steps")

        try:
            forecast = self.model.forecast(steps=forecast_steps)

            forecast = np.asarray(forecast)

            if len(forecast.shape) == 1:
                forecast = np.expand_dims(forecast, axis=1)

            self.logger.debug("ExponentialSmoothingModel", f"result shape: {forecast.shape}")
            return forecast

        except Exception as e:
            self.logger.error(f"Error during forecast: {e}")
            raise

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Anyvariate wrapper: predicts per variate and stacks columns.
        """
        if hasattr(self, "models") and self.models:
            if self.logger:
                self.logger.debug("ExponentialSmoothing Predict Wrapper", f"Multivariate prediction for {len(self.models)} variates")
            preds = []
            num_variates = len(self.models)
            for k, m in enumerate(self.models):
                if self.logger:
                    self.logger.debug("ExponentialSmoothing Predict Wrapper", f"Predicting for variate k={k}")
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(
                    y_context=yc,
                    timestamps_context=timestamps_context,
                    timestamps_target=timestamps_target,
                    freq=freq,
                    **kwargs,
                )
                preds.append(pk.reshape(-1, 1))
            result = np.concatenate(preds, axis=1)
            if self.logger:
                self.logger.debug("ExponentialSmoothing Predict Wrapper", f"Final concatenated prediction shape: {result.shape}")
                self.logger.info("ExponentialSmoothing Predict Wrapper", "Multivariate prediction completed successfully")
            return result
        # Univariate
        if self.logger:
            self.logger.debug("ExponentialSmoothing Predict Wrapper", "Univariate prediction")
        return self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            freq=freq,
            **kwargs,
        )
