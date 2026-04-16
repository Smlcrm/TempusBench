"""
TimesFM 2.0 500M foundation model for time series forecasting.

Uses the timesfm library with google/timesfm-2.0-500m-pytorch checkpoint.
The 500M model has no transformers port; it requires the timesfm package.
"""

import os
import numpy as np
import timesfm

from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs
from tempus_bench.models.timesfm_local_checkpoint import local_timesfm_checkpoint_file


class Timesfm500mHyperparams(PydanticBaseModel):
    pass


class Timesfm500mModel(BaseModel):
    """
    TimesFM 2.0 500M foundation model wrapper.

    Uses timesfm.TimesFm with google/timesfm-2.0-500m-pytorch.
    Returns stochastic samples from experimental quantile forecasts.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, Timesfm500mHyperparams)
        self._model = None

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
    ) -> "Timesfm500mModel":
        """Foundation model: no training needed. Build model and mark as fitted."""
        self._build_model()
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
        """
        Make predictions using TimesFM 500M. Returns stochastic samples
        (num_samples, forecast_horizon, num_targets) from quantile forecasts.
        Covariates supported via channel concatenation (past-only); covariate
        channel forecasts are discarded.
        """
        if not self.is_fitted:
            raise ValueError("Timesfm500mModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 100)

        # Channel concatenation: append x_context as extra channels when provided
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        n_channels = y_input.shape[1]
        inputs = [y_input[:, i].tolist() for i in range(n_channels)]
        point_forecast, quantile_forecast = self._model.forecast(
            inputs, freq=[0] * n_channels
        )

        # Keep only first num_targets channels (discard covariate channel forecasts)
        point_forecast = point_forecast[:num_targets]
        if quantile_forecast is not None and len(quantile_forecast) > 0:
            quantile_forecast = quantile_forecast[:num_targets]

        # point_forecast: list of (horizon,) arrays
        # quantile_forecast: list of (horizon, n_quantiles) or similar
        point_arr = np.array([np.asarray(x).flatten() for x in point_forecast])
        point_pred = point_arr.T[:forecast_horizon, :]  # (horizon, num_targets)

        # Build stochastic samples from quantiles if available
        if quantile_forecast is not None and len(quantile_forecast) > 0:
            q_arr = np.array([np.asarray(x) for x in quantile_forecast])
            # q_arr: (num_targets, horizon, n_quantiles) or (num_targets, horizon)
            if q_arr.ndim >= 3:
                n_quantiles = q_arr.shape[-1]
                # Truncate to forecast_horizon
                q_arr = q_arr[:, :forecast_horizon, :]
                # (num_targets, horizon, n_quantiles) -> (horizon, num_targets, n_quantiles)
                q_arr = np.transpose(q_arr, (1, 0, 2))
                if n_quantiles >= num_samples:
                    indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
                    samples = q_arr[:, :, indices]  # (horizon, num_targets, num_samples)
                else:
                    indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
                    indices = np.clip(indices, 0, n_quantiles - 1)
                    samples = q_arr[:, :, indices]
                # (horizon, num_targets, num_samples) -> (num_samples, horizon, num_targets)
                samples = np.transpose(samples, (2, 0, 1))
                return samples.astype(np.float64)

        # Fallback: repeat point forecast as pseudo-samples
        samples = np.tile(
            point_pred[np.newaxis, :, :], (num_samples, 1, 1)
        )
        return samples.astype(np.float64)

    def _build_model(self):
        device = getattr(self, "device", "cpu")
        context_len = int(getattr(self, "context_len", 2048))
        hf_or_dir = self.hf_model_name
        if os.path.isdir(hf_or_dir):
            ckpt_path = local_timesfm_checkpoint_file(hf_or_dir)
            hf_repo_id = None
        else:
            ckpt_path = None
            hf_repo_id = hf_or_dir
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=device,
                per_core_batch_size=32,
                input_patch_len=32,
                horizon_len=128,
                num_layers=50,
                model_dims=1280,
                output_patch_len=128,
                use_positional_embedding=False,
                context_len=context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path=ckpt_path,
                version="pytorch",
                huggingface_repo_id=hf_repo_id,
                local_dir=os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "checkpoints")
                ),
            ),
        )
