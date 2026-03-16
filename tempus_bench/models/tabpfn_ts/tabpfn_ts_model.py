"""
TabPFN-TS: Zero-shot time series forecasting with TabPFNv2.

Reframes forecasting as tabular regression. Supports point and probabilistic
forecasts with native covariate integration. Requires tabpfn-time-series.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
except ImportError as e:
    TabPFNTSPipeline = None
    TabPFNMode = None
    _TABPFN_TS_IMPORT_ERROR = str(e)


class TabpfnTsHyperparams(PydanticBaseModel):
    pass


class TabpfnTsModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, TabpfnTsHyperparams)
        if TabPFNTSPipeline is None:
            raise ImportError(
                "TabPFN-TS requires tabpfn-time-series. "
                f"Install with: pip install tabpfn-time-series. Original error: {_TABPFN_TS_IMPORT_ERROR}"
            )

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
    ) -> "TabpfnTsModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=True,
            model_name="TabPFN-TS",
        )
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
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=True,
            model_name="TabPFN-TS",
        )
        num_samples = kwargs.get("num_samples", 100)
        forecast_horizon = len(timestamps_target)
        num_variates = y_context.shape[1]

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)

        all_samples = []
        for v in range(num_variates):
            context_df = pd.DataFrame({
                "item_id": ["ts"] * len(timestamps_context),
                "timestamp": pd.to_datetime(timestamps_context, unit="ns"),
                "target": y_context[:, v].astype(np.float64),
            })
            # TabPFN-TS supports past-only and past+future. Future-only is not compatible
            # (schema mismatch: context has no covs but future would have covs).
            future_df = None
            if x_context is not None:
                for c in range(x_context.shape[1]):
                    context_df[f"cov_{c}"] = x_context[:, c].astype(np.float64)
            if x_context is not None and x_target is not None:
                # Both: add future covariates to future_df
                future_df = pd.DataFrame({
                    "item_id": ["ts"] * forecast_horizon,
                    "timestamp": pd.to_datetime(timestamps_target, unit="ns"),
                })
                x_target_trimmed = x_target[:forecast_horizon]
                for c in range(x_target_trimmed.shape[1]):
                    future_df[f"cov_{c}"] = x_target_trimmed[:, c].astype(np.float64)

            # predict_df requires exactly one of prediction_length or future_df
            if future_df is not None:
                pred_df = pipeline.predict_df(
                    context_df=context_df,
                    future_df=future_df,
                )
            else:
                pred_df = pipeline.predict_df(
                    context_df=context_df,
                    prediction_length=forecast_horizon,
                )
            if pred_df is None or pred_df.empty:
                samples = np.zeros((num_samples, forecast_horizon), dtype=np.float64)
            else:
                quant_cols = [c for c in pred_df.columns if c not in ("item_id", "timestamp")]
                if not quant_cols:
                    quant_cols = ["target"] if "target" in pred_df.columns else list(pred_df.columns)[2:]
                vals = pred_df[quant_cols].values
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)
                if vals.shape[1] >= num_samples:
                    samples = vals[:, :num_samples].T
                else:
                    indices = np.linspace(0, vals.shape[1] - 1, num_samples, dtype=int)
                    samples = vals[:, indices].T
            all_samples.append(samples)

        out = np.stack(all_samples, axis=-1)
        return out
