"""
Exponential Smoothing model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd

from itertools import product
from typing import Dict, Any, Union, Literal, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel
from tempus_bench.utils.logger import get_logger

class ExponentialSmoothingParams(PydanticBaseModel):
    trend: Optional[Literal["add", "mul"]] = Field(
        default=None, description="Trend component: 'add', 'mul', or None (no trend)"
    )
    seasonal: Optional[Literal["add", "mul"]] = Field(
        default=None, description="Seasonal component: 'add', 'mul', or None (no seasonality)"
    )
    seasonal_periods: Optional[int] = Field(
        default=None, ge=1, description="Number of seasonal periods (None if no seasonality)"
    )
    damped_trend: bool = Field(default=False, description="Whether to use damped trend")


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, config: JobConfig, logs_path: str):
        """
        Initialize Exponential Smoothing model with a given configuration.

        Args:
            config: JobConfig instance containing model and task configuration
            logs_path: Directory for storing log files (required)
        """
        super().__init__(config, logs_path, ExponentialSmoothingParams)
        self.model = None

    def get_hyperparameter_grid(self) -> list[dict]:
        """
        Override grid generation to enforce minimal cross-parameter constraints:
        - If seasonal is not None/'none', then seasonal_periods must be provided
        - If seasonal is None/'none', seasonal_periods must be None or 'none'
        - At least one of trend or seasonal must be specified (not None/'none')
        """
        params = self.model_config

        keys = list(params.keys())
        values_lists = [params[k] for k in keys]

        def _norm(x):
            if isinstance(x, str) and x.lower() == "none": return None
            return x

        def _is_valid_combination(c: dict) -> bool:
            trend = _norm(c["trend"])
            seasonal = _norm(c["seasonal"])
            seasonal_periods = c["seasonal_periods"]
            # require at least one of trend/seasonal
            if trend is None and seasonal is None: return False
            # consistency between seasonal and seasonal_periods
            if seasonal is not None and seasonal_periods is None: return False
            if seasonal is None and seasonal_periods is not None: return False
            # damped_trend only makes sense if trend present (handled in-fit but prefilter here)
            damped = c["damped_trend"]
            if damped and trend is None: return False
            return True

        grid: list[dict] = []
        for values_tuple in product(*values_lists):
            combo = dict(zip(keys, values_tuple))
            if _is_valid_combination(combo): grid.append(combo)
        return grid

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
        self.logger.debug("ExponentialSmoothingModel._train", f"y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")

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

        self.logger.debug("ExponentialSmoothingModel._train", f"endog shape: {endog.shape}, first 5 values: {endog[:5]}")
        self.logger.debug("ExponentialSmoothingModel._train", f"parameters: trend={trend}, seasonal={seasonal}, seasonal_periods={seasonal_periods}, damped_trend={damped_trend}")

        try:
            self.model = ExponentialSmoothing(
                endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend,
            ).fit()
            self.is_fitted = True
            self.logger.info("ExponentialSmoothingModel._train", "Model fitted successfully")
        except Exception as e:
            self.logger.error("ExponentialSmoothingModel._train", f"Error fitting model: {e}")
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

        Assumes y_context and y_target are 2D ndarrays: (num_steps, num_targets), even for univariate.
        """
        num_targets = y_context.shape[1]

        self.logger.debug("ExponentialSmoothingModel.train", f"Number of features/variates detected: {num_targets}")

        # Multivariate: more than one feature (column)
        if num_targets > 1:
            self.logger.debug("ExponentialSmoothingModel.train", "Taking multivariate path - training separate Exponential Smoothing per variate")
            self.models = []
            for k in range(num_targets):
                self.logger.debug("ExponentialSmoothingModel.train", f"Training variate k={k}")
                yc = y_context[:, k]    # Already 1D
                yt = y_target[:, k] if y_target is not None else None  # Already 1D
                # Create new model instance for this variate
                m = ExponentialSmoothingModel(self.config_path, logs_path=self.logs_path, hyperparameters=self.model_config)
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
            self.logger.info("ExponentialSmoothingModel.train", f"Multivariate training completed for {num_targets} variates")
            return self
        else:
            # Univariate: input is always (num_steps, 1)
            self.logger.debug("ExponentialSmoothingModel.train", "Taking univariate path")
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
        self.logger.debug("ExponentialSmoothingModel._predict", f"Forecasting {forecast_steps} steps")

        try:
            forecast = self.model.forecast(steps=forecast_steps)

            forecast = np.asarray(forecast)

            if len(forecast.shape) == 1:
                forecast = np.expand_dims(forecast, axis=1)

            self.logger.debug("ExponentialSmoothingModel._predict", f"result shape: {forecast.shape}")
            return forecast

        except Exception as e:
            self.logger.error("ExponentialSmoothingModel._predict", f"Error during forecast: {e}")
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
            self.logger.debug("ExponentialSmoothingModel.predict", f"Multivariate prediction for {len(self.models)} variates")
            preds = []
            num_variates = len(self.models)
            for k, m in enumerate(self.models):
                self.logger.debug("ExponentialSmoothingModel.predict", f"Predicting for variate k={k}")
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
            self.logger.debug("ExponentialSmoothingModel.predict", f"Final concatenated prediction shape: {result.shape}")
            self.logger.info("ExponentialSmoothingModel.predict", "Multivariate prediction completed successfully")
            return result
        # Univariate
        self.logger.debug("ExponentialSmoothingModel.predict", "Univariate prediction")
        return self._predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_target,
            freq=freq,
            **kwargs,
        )
