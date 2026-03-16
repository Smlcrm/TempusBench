from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from tempus_bench.models.base_model import BaseModel, validate_inputs


class NBEATSHyperparams(PydanticBaseModel):
    pass


class NBEATSModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, NBEATSHyperparams)
        self._nf = None
        self.is_fitted = False

    def _convert_to_datetimeindex(self, timestamps):
        timestamps = np.squeeze(timestamps)
        if isinstance(timestamps, pd.DatetimeIndex):
            return timestamps
        if isinstance(timestamps[0], (int, np.integer)):
            min_ts, max_ts = np.min(timestamps), np.max(timestamps)
            NS_LOWER, NS_UPPER = -9223372036854775808, 9223372036854775807
            US_LOWER, US_UPPER = NS_LOWER // 1000, NS_UPPER // 1000
            MS_LOWER, MS_UPPER = NS_LOWER // 1_000_000, NS_UPPER // 1_000_000
            S_LOWER, S_UPPER = NS_LOWER // 1_000_000_000, NS_UPPER // 1_000_000_000

            def in_bounds(val, lo, hi):
                return lo <= val <= hi

            if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(max_ts, NS_LOWER, NS_UPPER):
                unit = "ns"
            elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(max_ts, US_LOWER, US_UPPER):
                unit = "us"
            elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(max_ts, MS_LOWER, MS_UPPER):
                unit = "ms"
            elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(max_ts, S_LOWER, S_UPPER):
                unit = "s"
            else:
                raise ValueError(f"Timestamps out of bounds (min={min_ts}, max={max_ts})")
            timestamps = pd.to_datetime(timestamps, unit=unit)
        else:
            timestamps = pd.to_datetime(timestamps)
        return timestamps

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
    ) -> "NBEATSModel":
        freq = kwargs["freq"]
        forecast_horizon = y_target.shape[0]
        num_targets = y_context.shape[1]

        ds_context = self._convert_to_datetimeindex(timestamps_context)
        ds_target = self._convert_to_datetimeindex(timestamps_target)
        ds_all = ds_context.append(ds_target)

        dfs = []
        for t in range(num_targets):
            y_all = np.concatenate([y_context[:, t], y_target[:, t]])
            dfs.append(pd.DataFrame({
                "unique_id": f"target_{t}",
                "ds": ds_all,
                "y": y_all,
            }))

        train_df = pd.concat(dfs, ignore_index=True)

        input_size = min(self.input_size, y_context.shape[0])
        model = NBEATS(h=forecast_horizon, input_size=input_size, max_steps=self.max_steps)
        self._nf = NeuralForecast(models=[model], freq=freq)
        self._nf.fit(df=train_df)

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
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("NBEATSModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        ds_context = self._convert_to_datetimeindex(timestamps_context)

        dfs = []
        for t in range(num_targets):
            dfs.append(pd.DataFrame({
                "unique_id": f"target_{t}",
                "ds": ds_context,
                "y": y_context[:, t],
            }))

        context_df = pd.concat(dfs, ignore_index=True)
        preds = self._nf.predict(df=context_df)
        preds = preds.reset_index()

        forecasts = np.zeros((forecast_horizon, num_targets))
        for t in range(num_targets):
            target_preds = preds[preds["unique_id"] == f"target_{t}"]
            forecasts[:, t] = target_preds["NBEATS"].values[:forecast_horizon]

        return forecasts.astype(np.float64)
