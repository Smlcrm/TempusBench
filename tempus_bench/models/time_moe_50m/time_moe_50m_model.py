"""
Time-MoE mixture-of-experts foundation model for time series forecasting.

Time-MoE produces deterministic (point) forecasts via autoregressive generation.
It operates on univariate sequences, so multivariate targets are handled by
iterating over each target independently with mean/std normalization.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel
from transformers import AutoModelForCausalLM

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support


class TimeMoe50mHyperparams(PydanticBaseModel):
    pass


class TimeMoe50mModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, TimeMoe50mHyperparams)

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
    ) -> "TimeMoe50mModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Time-MoE",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            device_map=self.device,
            trust_remote_code=True,
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
            raise ValueError("TimeMoe50mModel is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Time-MoE",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        max_context = self.context_length - forecast_horizon
        if max_context <= 0:
            raise ValueError(
                f"forecast_horizon ({forecast_horizon}) exceeds context_length ({self.context_length})"
            )

        y_ctx = y_input[-max_context:, :]
        n_channels = y_ctx.shape[1]

        all_preds = []

        for i in range(n_channels):
            series = y_ctx[:, i].astype(np.float64)

            seqs = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
            mean = seqs.mean(dim=-1, keepdim=True)
            std = seqs.std(dim=-1, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            normed_seqs = (seqs - mean) / std

            with torch.no_grad():
                output = self._model.generate(normed_seqs, max_new_tokens=forecast_horizon)

            # output: (1, context_length + forecast_horizon)
            normed_preds = output[:, -forecast_horizon:]
            preds = normed_preds * std + mean

            all_preds.append(preds.squeeze(0).cpu().numpy())

        # Stack and keep only first num_targets (discard covariate channel forecasts)
        result = np.stack(all_preds, axis=-1)[:, :num_targets]
        return result.astype(np.float64)
