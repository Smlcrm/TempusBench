"""
Sundial diffusion-based foundation model for time series forecasting.

Sundial generates multiple sample trajectories via diffusion, making it a
stochastic forecaster. It operates on univariate sequences, so multivariate
targets are handled by iterating over each target independently with
z-score normalization.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel
from transformers import AutoModelForCausalLM

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support


def _patch_dynamic_cache_for_sundial():
    """Add seen_tokens/get_max_length aliases to DynamicCache for Sundial model compatibility."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        def get_max_length(self):
            shape = self.get_max_cache_shape()
            return shape if shape > 0 else 2**30

        DynamicCache.get_max_length = get_max_length


class SundialHyperparams(PydanticBaseModel):
    pass


class SundialModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, SundialHyperparams)

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
    ) -> "SundialModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Sundial",
        )
        _patch_dynamic_cache_for_sundial()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        self._model.eval()
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
            raise ValueError("SundialModel is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Sundial",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 20)
        lookback = getattr(self, "lookback_length", None) or y_context.shape[0]

        # Truncate to lookback_length (dev branch compatibility)
        y_ctx = y_context[-lookback:] if y_context.shape[0] > lookback else y_context

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            x_ctx = x_context[-lookback:] if x_context.shape[0] > lookback else x_context
            y_input = np.concatenate([y_ctx, x_ctx], axis=1)
        else:
            y_input = y_ctx

        n_channels = y_input.shape[1]
        all_samples = []

        for i in range(n_channels):
            series = y_input[:, i].astype(np.float64)

            mean = series.mean()
            std = series.std()
            if std == 0:
                std = 1.0
            normed = (series - mean) / std

            seqs = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = self._model.generate(
                    seqs, max_new_tokens=forecast_horizon, num_samples=num_samples
                )

            # output: (1, num_samples, context_length + forecast_horizon)
            preds = output[0, :, -forecast_horizon:].cpu().numpy()
            preds = preds * std + mean

            all_samples.append(preds)

        # all_samples: list of (num_samples, forecast_horizon), one per channel
        # Stack and keep only first num_targets (discard covariate channel forecasts)
        result = np.stack(all_samples, axis=-1)[:, :, :num_targets]
        return result.astype(np.float64)
