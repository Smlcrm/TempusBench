import numpy as np
import pandas as pd

from typing import Literal

from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from typing import Dict, Any, Union, Optional
from prophet.serialize import model_to_json, model_from_json
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel

class ProphetParams(PydanticBaseModel):
    seasonality_mode: Literal["additive", "multiplicative"] = Field(default="additive", description="Seasonality mode")
    yearly_seasonality: Optional[Union[int, bool]] = Field(default=None, description="Enable yearly seasonality (bool) or increase the number of Fourier terms (int)")
    weekly_seasonality: Optional[Union[int, bool]] = Field(default=None, description="Enable weekly seasonality (bool or increase the number of Fourier terms (int)")
    daily_seasonality: Optional[Union[int, bool]] = Field(default=None, description="Enable daily seasonality (bool) or increase the number of Fourier terms (int)")
    def __init__(self, **data):
        super().__init__(**data)
        # Enforce that only one of the seasonality options can be set (not None)
        seasonality_options = [
            self.yearly_seasonality,
            self.weekly_seasonality,
            self.daily_seasonality,
        ]
        set_options = [opt for opt in seasonality_options if opt is not None]
        if len(set_options) > 1:
            raise ValueError("Only one of yearly_seasonality, weekly_seasonality, or daily_seasonality can be set at the same time.")

class ProphetModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ProphetParams)
        self._build_model()
        self._scaler = StandardScaler()

    def _build_model(self):
        self._model = Prophet(**self.params.model_dump())
        self.is_fitted = False

    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ):
        y_context = y_context.squeeze()
        y_target = y_target.squeeze()
        timestamps_context = self._convert_to_datetimeindex(timestamps_context)
        timestamps_target = self._convert_to_datetimeindex(timestamps_target)

        train_df = pd.DataFrame({"ds": timestamps_context, "y": y_context})

        self._model.fit(train_df)
        self.is_fitted = True
        return self

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Generate predictions using the trained Prophet model.

        Args:
            y_context: Recent/past target values.
            timestamps_context: Timestamps for context data.
            timestamps_target: Timestamps for target/forecast steps.

        Returns:
            np.ndarray: Model predictions (num_steps, 1).
        """
        future_df = pd.DataFrame(
            {"ds": self._convert_to_datetimeindex(timestamps_target)}
        )

        forecast_df = self._model.predict(future_df)
        predictions = np.asarray(forecast_df["yhat"])
        if predictions.ndim == 1: predictions = np.expand_dims(predictions, axis=1)

        return predictions

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ):
        """
        Train the Prophet model. For multivariate data, fits one model per variate.

        Args:
            y_context: Training target values (shape: [n_samples] or [n_samples, n_variates])
            y_target: Optional target values (same shape as y_context)
            timestamps_context: Timestamps for training data
            timestamps_target: Timestamps for target data
            **kwargs: Additional arguments

        Returns:
            self
        """
        self._models = []
        for i in range(y_context.shape[1]):
            model = ProphetModel(params=self.params.dict())
            model._train(
                y_context=y_context[:, i],
                y_target=y_target[:, i] if y_target is not None else None,
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs
            )
            self._models.append(model)
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> np.ndarray:
        """
        Make predictions using the trained Prophet model(s).
        Handles both univariate and multivariate time series.

        Args:
            y_context: Recent/past target values.
            timestamps_context: Timestamps for context data.
            timestamps_target: Timestamps for target (forecast) data.

        Returns:
            np.ndarray: Model predictions (shape: [n_forecast_steps, n_variates])
        """
        if not self.is_fitted:
            raise ValueError("ProphetModel not fitted. Call train() first.")

        self._scaler.fit(y_context)
        y_scaled = self._scaler.transform(y_context)
        predictions = [
            model._predict(
                y_context=y_scaled[:, idx],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                **kwargs
            )
            for idx, model in enumerate(self._models)
        ]
        combined = np.column_stack(predictions)
        combined = self._scaler.inverse_transform(combined)
        return combined

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
