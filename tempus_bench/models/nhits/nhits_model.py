from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from tempus_bench.models.base_model import BaseModel, validate_inputs
from tempus_bench.models.neuralforecast_lightning_device import resolve_neuralforecast_trainer_kwargs


class NHITSHyperparams(PydanticBaseModel):
    """Tunable training knobs; aligned with TFT / shorter training."""

    model_config = ConfigDict(extra="forbid")

    input_size: int = Field(default=128, ge=1, description="NeuralForecast input_size (lookback).")
    max_steps: int = Field(
        default=25,
        ge=1,
        description="Lightning optimization steps per fit (not forecast horizon; tasks cap h at 128).",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Minibatch size (series per step); larger improves GPU/CPU throughput until memory-bound.",
    )


class NHITSModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, NHITSHyperparams)
        self._nf = None
        self._cov_names = []
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
    ) -> "NHITSModel":
        freq = kwargs["freq"]
        forecast_horizon = y_target.shape[0]
        num_targets = y_context.shape[1]

        ds_context = self._convert_to_datetimeindex(timestamps_context)
        ds_target = self._convert_to_datetimeindex(timestamps_target)
        ds_all = ds_context.append(ds_target)

        has_covariates = x_context is not None
        self._cov_names = []
        x_all = None
        if has_covariates:
            self._cov_names = [f"cov_{i}" for i in range(x_context.shape[1])]
            x_all = np.concatenate([x_context, x_target], axis=0)

        dfs = []
        for t in range(num_targets):
            y_all = np.concatenate([y_context[:, t], y_target[:, t]])
            df_data = {
                "unique_id": f"target_{t}",
                "ds": ds_all,
                "y": y_all,
            }
            if has_covariates:
                for c, name in enumerate(self._cov_names):
                    df_data[name] = x_all[:, c]
            dfs.append(pd.DataFrame(df_data))

        train_df = pd.concat(dfs, ignore_index=True)

        input_size = min(self.input_size, y_context.shape[0])
        trainer_kw = resolve_neuralforecast_trainer_kwargs(getattr(self, "device", None))
        val_check_steps = max(1, self.max_steps)
        if has_covariates:
            model = NHITS(
                h=forecast_horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                batch_size=self.batch_size,
                val_check_steps=val_check_steps,
                futr_exog_list=self._cov_names,
                hist_exog_list=self._cov_names,
                **trainer_kw,
            )
        else:
            model = NHITS(
                h=forecast_horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                batch_size=self.batch_size,
                val_check_steps=val_check_steps,
                **trainer_kw,
            )

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
            raise ValueError("NHITSModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        ds_context = self._convert_to_datetimeindex(timestamps_context)
        has_covariates = len(self._cov_names) > 0 and x_context is not None

        dfs = []
        for t in range(num_targets):
            df_data = {
                "unique_id": f"target_{t}",
                "ds": ds_context,
                "y": y_context[:, t],
            }
            if has_covariates:
                for c, name in enumerate(self._cov_names):
                    df_data[name] = x_context[:, c]
            dfs.append(pd.DataFrame(df_data))

        context_df = pd.concat(dfs, ignore_index=True)

        futr_df = None
        if has_covariates and x_target is not None:
            futr_df = self._nf.make_future_dataframe(df=context_df)
            x_target_trimmed = x_target[:forecast_horizon]
            for c, name in enumerate(self._cov_names):
                vals = np.tile(x_target_trimmed[:, c], num_targets)
                futr_df[name] = vals[:len(futr_df)]

        preds = self._nf.predict(df=context_df, futr_df=futr_df)
        preds = preds.reset_index()

        forecasts = np.zeros((forecast_horizon, num_targets))
        for t in range(num_targets):
            target_preds = preds[preds["unique_id"] == f"target_{t}"]
            forecasts[:, t] = target_preds["NHITS"].values[:forecast_horizon]

        return forecasts.astype(np.float64)
