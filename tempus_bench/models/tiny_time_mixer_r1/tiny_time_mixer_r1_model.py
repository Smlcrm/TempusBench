import numpy as np
import pandas as pd

from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from sktime.forecasting.ttm import TinyTimeMixerForecaster

from tempus_bench.models.base_model import BaseModel, validate_inputs


class TinyTimeMixerR1Hyperparams(PydanticBaseModel):
    pass


class TinyTimeMixerR1Model(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TinyTimeMixer model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, TinyTimeMixerR1Hyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TinyTimeMixerR1Model":
        """
        Train/fine-tune the foundation model on given data.

        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            timestamps_context: Timestamps for y_context (optional)
            timestamps_target: Timestamps for y_target (optional)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):
        num_targets = y_context.shape[1]
        forecast_horizon = timestamps_target.shape[0]
        columns = list(range(num_targets))
        timestamps_context = self._convert_to_datetimeindex(timestamps_context)
        df = pd.DataFrame(y_context, index=timestamps_context, columns=columns)
        fh = list(range(1, forecast_horizon + 1))
        model_path = getattr(self, "hf_model_name", None) or getattr(
            self, "model_path", "ibm/TTM"
        )
        revision = getattr(self, "revision", "main")
        self._model = TinyTimeMixerForecaster(
            model_path=model_path, revision=revision
        )

        X_fit = None
        X_pred = None
        timestamps_target_dt = self._convert_to_datetimeindex(timestamps_target)
        # TTM API requires X for both fit (past) and predict (future). Partial cases
        # use padding: past-only pads future with last row; future-only pads past with zeros.
        if x_context is not None and x_target is not None:
            cov_cols = list(range(x_context.shape[1]))
            X_fit = pd.DataFrame(x_context, index=timestamps_context, columns=cov_cols)
            x_target_trimmed = x_target[:forecast_horizon]
            X_pred = pd.DataFrame(
                x_target_trimmed, index=timestamps_target_dt, columns=cov_cols
            )
        elif x_context is not None:
            # Past-only: pad future with last observed value
            cov_cols = list(range(x_context.shape[1]))
            X_fit = pd.DataFrame(x_context, index=timestamps_context, columns=cov_cols)
            last_row = np.tile(x_context[-1:], (forecast_horizon, 1))
            X_pred = pd.DataFrame(
                last_row, index=timestamps_target_dt, columns=cov_cols
            )
        elif x_target is not None:
            # Future-only: pad past with zeros (unknown in past)
            x_target_trimmed = x_target[:forecast_horizon]
            cov_cols = list(range(x_target_trimmed.shape[1]))
            ctx_len = y_context.shape[0]
            past_pad = np.zeros((ctx_len, x_target_trimmed.shape[1]), dtype=np.float64)
            X_fit = pd.DataFrame(
                past_pad, index=timestamps_context, columns=cov_cols
            )
            X_pred = pd.DataFrame(
                x_target_trimmed, index=timestamps_target_dt, columns=cov_cols
            )

        self._model.fit(df, X=X_fit, fh=fh)
        forecast = self._model.predict(X=X_pred)
        return np.asarray(forecast)  # (forecast_horizon, num_targets)

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
