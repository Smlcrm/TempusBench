"""
Exponential Smoothing model implementation.
"""

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ExponentialSmoothingHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    trend: Optional[Literal["add", "mul", "null"]] = Field(
        ..., description="Trend component: 'add', 'mul', or None (no trend)"
    )
    damped_trend: bool = Field(..., description="Whether to use damped trend")
    # Fixed Hyperparameters - Optional for User to override
    seasonal: Optional[Literal["add", "mul", "null"]] = Field(
        default=None,
        description="Seasonal component: 'add', 'mul', or None (no seasonality)",
    )
    seasonal_periods: Optional[Union[int, Literal["null"]]] = Field(
        default=None,
        ge=1,
        description="Number of seasonal periods (None if no seasonality)",
    )

    @classmethod
    def validate(cls, value):
        # Only check *combinations* of values; field types/ranges handled by pydantic
        trend = value.get("trend")
        seasonal = value.get("seasonal")
        seasonal_periods = value.get("seasonal_periods")
        damped_trend = value.get("damped_trend")

        # 1. seasonal is set but seasonal_periods is not set
        if seasonal in ("add", "mul") and (
            seasonal_periods is None or seasonal_periods == "null"
        ):
            raise ValueError(
                "You must specify 'seasonal_periods' when using a seasonal component."
            )

        # 2. seasonal is None but seasonal_periods is set
        if (seasonal in (None, "null")) and (seasonal_periods not in (None, "null")):
            raise ValueError(
                "You cannot specify 'seasonal_periods' without a 'seasonal' component."
            )

        # 3. damped_trend=True but no trend
        if (trend in (None, "null")) and damped_trend:
            raise ValueError(
                "Cannot use 'damped_trend' without a trend component (trend=None/'null')."
            )

        return value

    def __init__(self, **data):
        # Handle string "null" values by converting them to None
        for k, v in data.items():
            if isinstance(v, str) and v.lower() == "null":
                data[k] = None
        super().__init__(**data)


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ExponentialSmoothingHyperparams)  # type: ignore
        self._models = []

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ):
        """
        Fit the Exponential Smoothing model on the provided context data.

        Args:
            y_context (np.ndarray): Historical target values for model fitting.
            y_target (np.ndarray): Future target values (unused; included for compatibility).
            timestamps_context (np.ndarray): Timestamps corresponding to y_context (unused).
            timestamps_target (np.ndarray): Timestamps corresponding to y_target (unused).

        Returns:
            The fitted ExponentialSmoothing model instance.

        Notes:
            Only y_context is used for fitting the Exponential Smoothing model. All other arguments are ignored to avoid data leakage.
        """
        trend = self.trend  # type: ignore
        seasonal = self.seasonal  # type: ignore
        seasonal_periods = self.seasonal_periods  # type: ignore
        damped_trend = self.damped_trend  # type: ignore

        # Cannot use damped_trend without a trend component
        if trend is None or trend == "null":
            damped_trend = False

        endog = y_context.squeeze()

        fitted_model = ExponentialSmoothing(
            endog=endog,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
        ).fit()

        return fitted_model

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> "ExponentialSmoothingModel":
        """
        Trains a separate Exponential Smoothing model for each variate in a (potentially multivariate) time series.

        Expects y_context and y_target as 2D arrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]

        for idx in range(num_targets):
            fitted_model = self._train(
                y_context=y_context[:, idx : idx + 1],
                y_target=y_target[:, idx : idx + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs,
            )
            self._models.append(fitted_model)

        self.is_fitted = True

        return self

    @validate_inputs
    def _predict(
        self,
        fitted_model,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predicts future values using the trained Exponential Smoothing model.

        Args:
            fitted_model: The fitted ExponentialSmoothing model instance.
            y_context: Past target values (unused, present for interface consistency)
            timestamps_context: Context timestamps (unused)
            timestamps_target: Timestamps for the forecast horizon

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If the model is not fitted or forecast horizon is invalid.
        """
        if not self.is_fitted:
            raise ValueError(
                "ExponentialSmoothingModel not fitted. Call train() first."
            )

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Exponential Smoothing prediction."
            )

        forecast_steps = len(timestamps_target)
        forecast = fitted_model.forecast(steps=forecast_steps)
        y_pred = np.asarray(forecast).reshape(-1, 1)

        return y_pred

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predicts future values for each variate and concatenates the results.

        Args:
            y_context (np.ndarray): Context values, shape (num_steps, num_variates).
            timestamps_context (np.ndarray): Timestamps for context data.
            timestamps_target (np.ndarray): Timestamps for target/future data.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, num_variates).

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError(
                "ExponentialSmoothingModel not fitted. Call train() first."
            )

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                fitted_model=fitted_model,
                y_context=y_context[:, idx : idx + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs,
            )
            preds.append(prediction)

        result = np.concatenate(preds, axis=-1)
        return result
