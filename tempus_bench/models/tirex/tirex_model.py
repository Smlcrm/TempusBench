"""
TiRex foundation model implementation for time series forecasting.

TiRex is a channel-independent transformer model with a covariate encoder.
It produces quantile forecasts which are expanded into pseudo-samples for
compatibility with the stochastic evaluation pipeline.

Requires tirex-ts>=1.4.0.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tirex import load_model, ForecastModel
except ImportError as e:
    load_model = None
    ForecastModel = None
    _TIREX_IMPORT_ERROR = str(e)


class TiRexHyperparams(PydanticBaseModel):
    pass


class TiRexModel(BaseModel):
    """TiRex zero-shot forecasting model."""

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, TiRexHyperparams)
        self.backend = getattr(self, "backend", "torch")
        if load_model is None:
            raise ImportError(
                "TiRex requires tirex-ts>=1.4.0. "
                f"Install with: pip install 'tirex-ts>=1.4.0'. "
                f"Original error: {_TIREX_IMPORT_ERROR}"
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
    ) -> "TiRexModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="TiRex",
        )
        self._model = load_model(self.hf_model_name, backend=self.backend)
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
            supports_both=False,
            model_name="TiRex",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 100)

        # Past covariates via channel concatenation (targets + covariates as extra channels)
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        # TiRex expects context (n_channels, context_length)
        context = torch.tensor(y_input.T, dtype=torch.float32)
        quantiles, mean = self._model.forecast(
            context=context,
            prediction_length=forecast_horizon,
        )

        if isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()

        # quantiles: (n_channels, forecast_len, n_quantiles); keep only target channels
        quantiles = quantiles[:num_targets]
        all_samples = []
        for t in range(num_targets):
            q = quantiles[t].T  # (n_quantiles, forecast_len)
            n_quantiles = q.shape[0]
            if n_quantiles >= num_samples:
                samples_t = q[:num_samples, :]
            else:
                indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
                samples_t = q[indices, :]
            all_samples.append(samples_t)

        stacked = np.stack(all_samples, axis=0)
        result = np.transpose(stacked, (1, 2, 0))
        return result.astype(np.float64)


# Alias for model executor (derives TirexModel from model_name "tirex")
TirexModel = TiRexModel
