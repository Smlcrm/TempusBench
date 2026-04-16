"""
TimesFM 2.5 foundation model implementation for time series forecasting.

This module provides a wrapper around Google's TimesFM 2.5 via Hugging Face
transformers. TimesFM 2.5 supports:
- Stochastic output: continuous quantile forecasts (converted to samples)
- Multivariate: iterates over targets (channel-independent)
- Non-autoregressive: predicts full horizon in one forward pass
- Covariates: extended-input support (x_context only) via channel concatenation

Requires transformers>=5.3.0 with TimesFm2_5ModelForPrediction.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from transformers import TimesFm2_5ModelForPrediction
except ImportError as e:
    TimesFm2_5ModelForPrediction = None
    _TIMESFM2_IMPORT_ERROR = str(e)


class Timesfm2Hyperparams(PydanticBaseModel):
    pass


class Timesfm2Model(BaseModel):
    """
    TimesFM 2.5 foundation model wrapper with stochastic (quantile) output.

    Uses the transformers TimesFm2_5ModelForPrediction which returns both
    mean_predictions and full_predictions (quantiles). Quantiles are converted
    to pseudo-samples for stochastic metrics (CRPS, etc.).
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, Timesfm2Hyperparams)
        if TimesFm2_5ModelForPrediction is None:
            raise ImportError(
                "TimesFM 2.5 requires transformers>=4.46.0 with TimesFm2_5 support. "
                f"Install with: pip install 'transformers>=5.3.0'. Original error: {_TIMESFM2_IMPORT_ERROR}"
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
    ) -> "Timesfm2Model":
        """
        Foundation model: no training needed. Build model and mark as fitted.
        Covariates (x_context, x_target) are ignored; TimesFM 2.5 has no native covariate support.
        """
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
        Make predictions using TimesFM 2.5. Returns stochastic samples
        (num_samples, forecast_horizon, num_targets) from quantile forecasts.
        Covariates are ignored.
        """
        if not self.is_fitted:
            raise ValueError("Timesfm2Model is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="TimesFM 2.5",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 100)

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        # Trim or pad context to context_length
        context_length = min(self.context_length, y_input.shape[0])
        if y_input.shape[0] >= context_length:
            y_trimmed = y_input[-context_length:, :]
        else:
            pad_rows = context_length - y_input.shape[0]
            y_trimmed = np.pad(y_input, ((pad_rows, 0), (0, 0)), mode="edge")

        # TimesFM 2.5 expects Sequence of 1D tensors: one per series
        n_channels = y_trimmed.shape[1]
        device = next(self._model.parameters()).device
        past_values = [
            torch.tensor(y_trimmed[:, i].astype(np.float32), device=device)
            for i in range(n_channels)
        ]

        with torch.no_grad():
            outputs = self._model(past_values=past_values, return_dict=True)

        mean_pred = outputs.mean_predictions.cpu().numpy()  # (batch, horizon)
        full_pred = outputs.full_predictions.cpu().numpy()  # (batch, horizon, quantiles)

        # Model outputs fixed horizon (default 128); truncate or pad to forecast_horizon
        model_horizon = full_pred.shape[1]
        if model_horizon >= forecast_horizon:
            full_pred = full_pred[:, :forecast_horizon, :]
        else:
            # Pad by repeating last step
            pad_width = forecast_horizon - model_horizon
            last_step = full_pred[:, -1:, :]  # (n_channels, 1, n_quantiles)
            pad_block = np.tile(last_step, (1, pad_width, 1))
            full_pred = np.concatenate([full_pred, pad_block], axis=1)

        # Keep only first num_targets channels (discard covariate channel forecasts)
        full_pred = full_pred[:num_targets, :, :]
        n_targets_out, horizon_out, n_quantiles = full_pred.shape

        # Convert quantiles to samples for stochastic metrics
        if n_quantiles >= num_samples:
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = full_pred[:, :, indices]  # (n_targets, horizon, num_samples)
        else:
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = full_pred[:, :, indices]

        # Target shape: (num_samples, forecast_horizon, num_targets)
        samples = np.transpose(samples, (2, 1, 0))
        return samples.astype(np.float64)

    def _build_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = "auto" if device == "cuda" else None

        self._model = TimesFm2_5ModelForPrediction.from_pretrained(
            self.hf_model_name,
            device_map=device_map or device,
        )
        self._model.eval()
