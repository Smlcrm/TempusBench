"""
Exponential Smoothing model implementation.
"""
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ...base_model import BaseModel, validate_inputs

class ExponentialSmoothingHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    trend: Optional[Literal["add", "mul", "null"]] = Field(
        ..., description="Trend component: 'add', 'mul', or None (no trend)"
    )
    damped_trend: bool = Field(..., description="Whether to use damped trend")
    ### Fixed Hyperparameters - Optional for User to override
    seasonal: Optional[Literal["add", "mul", "null"]] = Field(
        default=None, description="Seasonal component: 'add', 'mul', or None (no seasonality)"
    )
    seasonal_periods: Optional[Union[int, Literal["null"]]] = Field(
        default=None, ge=1, description="Number of seasonal periods (None if no seasonality)"
    )

    @classmethod
    def validate(cls, value):
        # Only check *combinations* of values; field types/ranges handled by pydantic
        trend = value.get('trend')
        seasonal = value.get('seasonal')
        seasonal_periods = value.get('seasonal_periods')
        damped_trend = value.get('damped_trend')

        # 1. seasonal is set but seasonal_periods is not set
        if seasonal in ("add", "mul") and (seasonal_periods is None or seasonal_periods == "null"):
            raise ValueError("You must specify 'seasonal_periods' when using a seasonal component.")

        # 2. seasonal is None but seasonal_periods is set
        if (seasonal in (None, "null")) and (seasonal_periods not in (None, "null")):
            raise ValueError("You cannot specify 'seasonal_periods' without a 'seasonal' component.")

        # 3. damped_trend=True but no trend
        if (trend in (None, "null")) and damped_trend:
            raise ValueError("Cannot use 'damped_trend' without a trend component (trend=None/'null').")

        return value

    def __init__(self, **data):
        # Handle string "null" values by converting them to None
        for k, v in data.items():
            if isinstance(v, str) and v.lower() == "null":
                data[k] = None
        super().__init__(**data)


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ExponentialSmoothingHyperparams)

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "ExponentialSmoothingModel":
        """
        Train a single Exponential Smoothing model on univariate data.
        """
        self.logger.debug(
            "ExponentialSmoothingModel._train",
            f"y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}"
        )

        trend = self.trend
        seasonal = self.seasonal
        seasonal_periods = self.seasonal_periods
        damped_trend = self.damped_trend

        endog = y_context.squeeze()

        self.logger.debug(
            "ExponentialSmoothingModel._train",
            f"endog shape: {endog.shape}, first 5 values: {endog[:5]}"
        )
        self.logger.debug(
            "ExponentialSmoothingModel._train",
            (
                f"parameters: trend={trend}, seasonal={seasonal}, "
                f"seasonal_periods={seasonal_periods}, damped_trend={damped_trend}"
            )
        )
        try:
            self._model = ExponentialSmoothing(
                endog=endog,
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

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "ExponentialSmoothingModel":
        """
        Trains a separate Exponential Smoothing model for each variate in the input data.

        Assumes y_context and y_target are 2D ndarrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]
        self.logger.debug("ExponentialSmoothingModel.train", f"Number of variates detected: {num_targets}")

        self._models = []
        for idx in range(num_targets):
            self.logger.debug("ExponentialSmoothingModel.train", f"Training variate index {idx}")
            model = ExponentialSmoothingModel(params=self.dict())
            model._train(
                y_context=y_context[:, idx],
                y_target=y_target[:, idx],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs,
            )
            self._models.append(model)
        self.is_fitted = True
        self.logger.info(
            "ExponentialSmoothingModel.train",
            f"Training completed for {num_targets} variate(s)."
        )
        return self

    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Predict using a single Exponential Smoothing model on univariate data.
        """
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")

        forecast_steps = len(timestamps_target)
        self.logger.debug(
            "ExponentialSmoothingModel._predict",
            f"Forecasting {forecast_steps} steps"
        )

        try:
            forecast = np.asarray(self._model.forecast(steps=forecast_steps))
            if forecast.ndim == 1: forecast = forecast[:, np.newaxis]
            self.logger.debug(
                "ExponentialSmoothingModel._predict",
                f"Result shape: {forecast.shape}"
            )
            return forecast

        except Exception as e:
            self.logger.error(
                "ExponentialSmoothingModel._predict",
                f"Error during forecast: {e}"
            )
            raise

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Wrapper to predict per variate and stack columns for multivariate data.
        """
        self.logger.debug(
            "ExponentialSmoothingModel.predict",
            f"Multivariate prediction for {len(self._models)} variates"
        )
        preds = []
        for target_idx, model in enumerate(self._models):
            self.logger.debug(
                "ExponentialSmoothingModel.predict",
                f"Predicting for variate index {target_idx}"
            )
            pred = model._predict(
                y_context=y_context[:, target_idx],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs
            )
            preds.append(pred.reshape(-1, 1))
        result = np.concatenate(preds, axis=1)
        self.logger.debug(
            "ExponentialSmoothingModel.predict",
            f"Final concatenated prediction shape: {result.shape}"
        )
        self.logger.info(
            "ExponentialSmoothingModel.predict",
            "Multivariate prediction completed successfully"
        )
        return result
