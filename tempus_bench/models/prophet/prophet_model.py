from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel as PydanticBaseModel, Field, model_validator

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ProphetHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    growth: Literal["linear", "logistic", "flat"] = Field(
        default="linear", description="Growth type"
    )
    seasonality_mode: Literal["additive", "multiplicative"] = Field(
        ..., description="Seasonality mode"
    )
    changepoint_prior_scale: float = Field(
        ..., ge=0, le=1, description="Changepoint prior scale"
    )
    seasonality_prior_scale: float = Field(
        ..., ge=0, description="Seasonality prior scale"
    )
    # Fixed Hyperparameters - Optional for User to override
    yearly_seasonality: Optional[Union[int, bool]] = Field(
        default=False,
        description="Enable yearly seasonality (bool) or increase the number of Fourier terms (int)",
    )
    weekly_seasonality: Optional[Union[int, bool]] = Field(
        default=False,
        description="Enable weekly seasonality (bool) or increase the number of Fourier terms (int)",
    )
    daily_seasonality: Optional[Union[int, bool]] = Field(
        default=False,
        description="Enable daily seasonality (bool) or increase the number of Fourier terms (int)",
    )

    @model_validator(mode='after')
    def validate_combinations(self):
        # Enforce that only one of the seasonality options can be enabled (truthy)
        seasonality_options = [
            self.yearly_seasonality,
            self.weekly_seasonality,
            self.daily_seasonality,
        ]
        # Check for truthy values (True or positive integers), not just "not None"
        # Default values are False, which is fine - only count truthy values
        enabled_options = [opt for opt in seasonality_options if opt]
        if len(enabled_options) > 1:
            raise ValueError(
                "Only one of yearly_seasonality, weekly_seasonality, or daily_seasonality can be enabled at the same time."
            )
        return self


class ProphetModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ProphetHyperparams)
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
        Fit the Prophet model on the provided context data.

        Args:
            y_context (np.ndarray): Historical target values for model fitting.
            y_target (np.ndarray): Future target values (unused; included for compatibility).
            timestamps_context (np.ndarray): Timestamps corresponding to y_context.
            timestamps_target (np.ndarray): Timestamps corresponding to y_target (unused).

        Returns:
            The fitted Prophet model instance.

        Notes:
            Only y_context is used for fitting the Prophet model. All other arguments are ignored to avoid data leakage.
        """
        y_context = y_context.squeeze()
        timestamps_context = self._convert_to_datetimeindex(timestamps_context)

        train_df = pd.DataFrame({"ds": timestamps_context, "y": y_context})

        # Construct model_params dict: {param_name: value for valid Prophet parameters}
        model_params = {
            k: getattr(self, k)
            for k in self.params_class.model_fields.keys()
            if hasattr(self, k)
        }
        model = Prophet(**model_params)
        fitted_model = model.fit(train_df)

        return fitted_model

    @validate_inputs
    def _predict(
        self,
        prophet_model,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Predicts future values using the trained Prophet model.

        Args:
            prophet_model: The fitted Prophet model instance.
            y_context: Past target values (unused, present for interface consistency)
            timestamps_context: Context timestamps (unused)
            timestamps_target: Timestamps for the forecast horizon

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, 1)

        Raises:
            ValueError: If the model is not fitted or forecast horizon is invalid.
        """

        future_df = pd.DataFrame(
            {"ds": self._convert_to_datetimeindex(timestamps_target)}
        )

        forecast_df = prophet_model.predict(future_df)
        predictions = np.asarray(forecast_df["yhat"])
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=1)

        return predictions

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> "ProphetModel":
        """
        Trains a separate Prophet model for each variate in a (potentially multivariate) time series.

        Expects y_context and y_target as 2D arrays: (num_steps, num_targets).
        """
        num_targets = y_context.shape[1]

        for k in range(num_targets):
            fitted_model = self._train(
                y_context=y_context[:, k : k + 1],
                y_target=y_target[:, k : k + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs,
            )
            self._models.append(fitted_model)

        self.is_fitted = True

        return self

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
            raise ValueError("ProphetModel not fitted. Call train() first.")

        preds = []
        for idx, prophet_model in enumerate(self._models):
            prediction = self._predict(
                prophet_model=prophet_model,
                y_context=y_context[:, idx : idx + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs,
            )
            preds.append(prediction)

        result = np.concatenate(preds, axis=-1)
        return result

    def _convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                min_ts = np.min(timestamps)
                max_ts = np.max(timestamps)

                # Pandas datetime bounds for 64-bit ns: 1677-09-21 to 2262-04-11
                # 1677-09-21T00:12:43.145224192Z = -9223372036854775808 ns
                # 2262-04-11T23:47:16.854775807Z = 9223372036854775807 ns
                NS_LOWER = -9223372036854775808
                NS_UPPER = 9223372036854775807
                US_LOWER = NS_LOWER // 1000
                US_UPPER = NS_UPPER // 1000
                MS_LOWER = NS_LOWER // 1_000_000
                MS_UPPER = NS_UPPER // 1_000_000
                S_LOWER = NS_LOWER // 1_000_000_000
                S_UPPER = NS_UPPER // 1_000_000_000

                def in_bounds(val, lower, upper):
                    return lower <= val <= upper

                # Try to classify the likely unit and check bounds
                unit = None
                if isinstance(min_ts, (int, np.integer)):
                    # Try nanoseconds
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(
                        max_ts, NS_LOWER, NS_UPPER
                    ):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(
                        max_ts, US_LOWER, US_UPPER
                    ):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(
                        max_ts, MS_LOWER, MS_UPPER
                    ):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(
                        max_ts, S_LOWER, S_UPPER
                    ):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps
