"""
Moirai 2.0 foundation model for time series forecasting.

Moirai 2.0 is a decoder-only universal time series forecasting transformer from Salesforce.
It uses quantile loss and multi-token prediction. Supports past and future covariates.

Requires uni2ts with Moirai 2.0 support.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs

try:
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
except ImportError as e:
    Moirai2Forecast = None
    Moirai2Module = None
    _MOIRAI2_IMPORT_ERROR = str(e)


class Moirai2Hyperparams(PydanticBaseModel):
    pass


class Moirai2Model(BaseModel):
    """
    Moirai 2.0 foundation model wrapper.

    Moirai 2.0 is decoder-only, uses quantile forecasts, and supports
    past + future covariates. Native multivariate.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, Moirai2Hyperparams)
        if Moirai2Forecast is None:
            raise ImportError(
                "Moirai 2.0 requires uni2ts with Moirai 2.0 support. "
                f"Install from: pip install uni2ts. Original error: {_MOIRAI2_IMPORT_ERROR}"
            )
        self._model = None
        self.is_fitted = False

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
    ) -> "Moirai2Model":
        """Initialize Moirai 2.0 (no training required for foundation models)."""
        if not self.is_fitted:
            train_span = int(y_target.shape[0])
            val_h_raw = kwargs.get("validate_horizon")
            val_h = int(val_h_raw) if val_h_raw is not None else train_span
            pdt = max(train_span, val_h)
            ctx_train = y_context.shape[0] + y_target.shape[0]
            # Moirai2 uses hparams.context_length for token counts; predict receives
            # y_context = context+train (full history). Use that length so observed_mask
            # and prediction_mask shapes match in _convert.
            # Moirai2 has known issues with feat_dynamic_real; use past_feat_dynamic_real only
            past_feat_dim = 0
            if x_context is not None or x_target is not None:
                past_feat_dim = (
                    x_context.shape[1] if x_context is not None else x_target.shape[1]
                )

            module = Moirai2Module.from_pretrained(
                pretrained_model_name_or_path=self.hf_model_name
            )
            patch_size = getattr(module, "patch_size", 16)
            # Moirai2 requires context_length and prediction_length divisible by patch_size
            ctx_aligned = ((ctx_train + patch_size - 1) // patch_size) * patch_size
            pdt_aligned = ((pdt + patch_size - 1) // patch_size) * patch_size
            self._model = Moirai2Forecast(
                module=module,
                prediction_length=pdt_aligned,
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=past_feat_dim,
                context_length=ctx_aligned,
            )
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()

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
        """Make predictions using Moirai 2.0. Returns samples (num_samples, forecast_horizon, num_targets)."""
        if not self.is_fitted:
            raise ValueError("Moirai2Model is not fitted. Call train() first.")

        num_targets = y_context.shape[1]
        ctx = y_context.shape[0]
        forecast_horizon = timestamps_target.shape[0]
        num_samples_raw = kwargs.get("num_samples")
        if num_samples_raw is None:
            raise ValueError("num_samples is required for Moirai2Model.predict")
        num_samples = int(num_samples_raw)
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        # Use the model's stored context_length (set during train), not a recomputed value.
        # predict() receives y_context = target[cstart:tend] which includes context+train steps,
        # but the model was initialized with context_length = ceil(train_ctx / patch_size)*patch_size.
        # Truncating to model_context_length avoids the observed_mask vs prediction_mask mismatch.
        model_context_length = self._model.hparams.context_length

        # Align y_context to model_context_length: take the last model_context_length steps
        if ctx > model_context_length:
            y_context = y_context[-model_context_length:, :]
            if x_context is not None:
                x_context = x_context[-model_context_length:, :]
            ctx = model_context_length
        elif ctx < model_context_length:
            pad_rows = model_context_length - ctx
            y_context = np.pad(y_context, ((pad_rows, 0), (0, 0)), mode="edge")
            if x_context is not None:
                x_context = np.pad(x_context, ((pad_rows, 0), (0, 0)), mode="edge")
            ctx = model_context_length

        observed_mask = np.ones((ctx, num_targets), dtype=bool)
        past_target = torch.tensor(y_context, dtype=torch.float32).unsqueeze(0)
        past_observed_target = torch.tensor(observed_mask, dtype=torch.bool).unsqueeze(0)
        past_is_pad = (~torch.tensor(observed_mask, dtype=torch.bool)).any(dim=-1).unsqueeze(0)

        device = next(self._model.parameters()).device
        past_target = past_target.to(device)
        past_observed_target = past_observed_target.to(device)
        past_is_pad = past_is_pad.to(device)

        feat_dynamic_real = None
        observed_feat_dynamic_real = None
        past_feat_dynamic_real = None
        past_observed_feat_dynamic_real = None

        # Moirai2: use past_feat_dynamic_real only (feat_dynamic_real has known bugs).
        # past_feat_dynamic_real must match context_length (ctx_aligned).
        past_feat_dim = getattr(self._model, "past_feat_dynamic_real_dim", 0)
        if past_feat_dim > 0:
            if x_context is not None:
                past_feat_arr = x_context
            else:
                past_feat_arr = np.zeros((ctx, past_feat_dim), dtype=np.float32)
            past_feat_dynamic_real = torch.tensor(
                past_feat_arr, dtype=torch.float32
            ).unsqueeze(0).to(device)
            past_observed_feat_dynamic_real = torch.ones_like(
                past_feat_dynamic_real, dtype=torch.bool
            )

        with torch.no_grad():
            pred = self._model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                feat_dynamic_real=None,
                observed_feat_dynamic_real=None,
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
            )

        # Forward returns (batch, num_quantiles, prediction_length) for univariate
        # or (batch, num_quantiles, prediction_length, num_targets) multivariate.
        pred = pred.cpu().numpy()
        if pred.ndim >= 1 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        # Truncate to requested forecast_horizon (model may predict pdt_aligned)
        if pred.ndim == 2:
            # (num_quantiles, prediction_length)
            pred = pred[:, :forecast_horizon]
            pred = pred[..., np.newaxis]
        else:
            pred = pred[:, :forecast_horizon, :]
        # pred: (num_quantiles, forecast_horizon, num_targets)
        n_quantiles = pred.shape[0]

        # Convert quantiles to pseudo-samples
        if n_quantiles >= num_samples:
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = pred[indices, :, :]
        else:
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = pred[np.clip(indices, 0, n_quantiles - 1), :, :]

        # samples: (num_samples, prediction_length, num_targets)
        return samples.astype(np.float64)
