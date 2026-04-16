from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs
from tempus_bench.models.neuralforecast_lightning_device import resolve_neuralforecast_trainer_kwargs

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
except ImportError as e:
    NeuralForecast = None
    TFT = None
    _NF_IMPORT_ERROR = str(e)


class TFTHyperparams(PydanticBaseModel):
    """Tunable training knobs; defaults favor shorter wall-clock on GPU Batch jobs."""

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
        description="Minibatch size (series per step); larger improves GPU throughput until memory-bound.",
    )


class TFTModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, TFTHyperparams)
        if NeuralForecast is None:
            raise ImportError(
                "TFT requires neuralforecast>=2.0.0. "
                f"Install with: pip install 'neuralforecast>=2.0.0'. Original error: {_NF_IMPORT_ERROR}"
            )
        self._nf = None

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
    ) -> "TFTModel":
        freq = kwargs["freq"]
        forecast_horizon = y_target.shape[0]
        input_size = min(self.input_size, y_context.shape[0])
        num_targets = y_context.shape[1]

        ts_context = self._convert_to_datetimeindex(timestamps_context)
        ts_target = self._convert_to_datetimeindex(timestamps_target)
        all_timestamps = ts_context.append(ts_target)

        has_covariates = x_context is not None and x_target is not None
        if has_covariates:
            num_covariates = x_context.shape[1]
            cov_names = [f"cov_{i}" for i in range(num_covariates)]
            x_all = np.concatenate([x_context, x_target], axis=0)

        frames = []
        for k in range(num_targets):
            y_all = np.concatenate([y_context[:, k], y_target[:, k]], axis=0)
            df = pd.DataFrame({
                "unique_id": f"target_{k}",
                "ds": all_timestamps,
                "y": y_all,
            })
            if has_covariates:
                for i, name in enumerate(cov_names):
                    df[name] = x_all[:, i]
            frames.append(df)

        train_df = pd.concat(frames, ignore_index=True)

        trainer_kw = resolve_neuralforecast_trainer_kwargs(getattr(self, "device", None))
        # One validation per fit cuts Lightning overhead on short max_steps runs.
        val_check_steps = max(1, self.max_steps)
        if has_covariates:
            model = TFT(
                h=forecast_horizon,
                input_size=input_size,
                max_steps=self.max_steps,
                batch_size=self.batch_size,
                val_check_steps=val_check_steps,
                futr_exog_list=cov_names,
                hist_exog_list=cov_names,
                **trainer_kw,
            )
        else:
            model = TFT(
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
            raise ValueError("TFTModel is not fitted. Call train() first.")

        num_targets = y_context.shape[1]
        ts_context = self._convert_to_datetimeindex(timestamps_context)
        ts_target = self._convert_to_datetimeindex(timestamps_target)

        has_covariates = x_context is not None and x_target is not None
        if has_covariates:
            num_covariates = x_context.shape[1]
            cov_names = [f"cov_{i}" for i in range(num_covariates)]

        context_frames = []
        for k in range(num_targets):
            df = pd.DataFrame({
                "unique_id": f"target_{k}",
                "ds": ts_context,
                "y": y_context[:, k],
            })
            if has_covariates:
                for i, name in enumerate(cov_names):
                    df[name] = x_context[:, i]
            context_frames.append(df)
        context_df = pd.concat(context_frames, ignore_index=True)

        futr_df = None
        if has_covariates:
            futr_frames = []
            for k in range(num_targets):
                df = pd.DataFrame({
                    "unique_id": f"target_{k}",
                    "ds": ts_target,
                })
                for i, name in enumerate(cov_names):
                    df[name] = x_target[:, i]
                futr_frames.append(df)
            futr_df = pd.concat(futr_frames, ignore_index=True)

        if futr_df is not None:
            preds_df = self._nf.predict(df=context_df, futr_df=futr_df)
        else:
            preds_df = self._nf.predict(df=context_df)

        preds_df = preds_df.reset_index()
        forecast_horizon = timestamps_target.shape[0]
        result = np.zeros((forecast_horizon, num_targets), dtype=np.float64)
        for k in range(num_targets):
            mask = preds_df["unique_id"] == f"target_{k}"
            result[:, k] = preds_df.loc[mask, "TFT"].values[:forecast_horizon]

        return result

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
