"""
TiRex foundation model implementation for time series forecasting.

TiRex is a channel-independent transformer model with a covariate encoder.
It produces quantile forecasts which are expanded into pseudo-samples for
compatibility with the stochastic evaluation pipeline.

Requires tirex-ts>=1.4.0.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tirex import load_model, ForecastModel
    from tirex.base import xlstm_available
except ImportError as e:
    load_model = None
    ForecastModel = None
    xlstm_available = None
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

    def _resolve_tirex_checkpoint_path(self, path: str) -> str:
        """Local dir uses ``model.ckpt``; ``tirex.load_model`` breaks on deep POSIX paths."""
        if os.path.isdir(path):
            ckpt = os.path.join(path, "model.ckpt")
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    f"Expected TiRex checkpoint at {ckpt!r} (directory {path!r})"
                )
            return ckpt
        return path

    def _load_tirex_model(self) -> Any:
        """Load from hub id (e.g. NX-AI/TiRex) or local checkpoint dir / file."""
        path = self.hf_model_name
        ckpt_path: str | None = None
        if os.path.isdir(path):
            ckpt_path = self._resolve_tirex_checkpoint_path(path)
        elif os.path.isfile(path):
            ckpt_path = path

        if ckpt_path is not None:
            from tirex.base import PretrainedModel

            register_key = self.settings.get("tirex_register_name")
            if not isinstance(register_key, str) or not register_key.strip():
                register_key = "TiRex"
            model_cls = PretrainedModel.REGISTRY.get(register_key)
            if model_cls is None:
                raise ValueError(
                    f"Unknown tirex_register_name={register_key!r}; "
                    f"valid keys include {sorted(PretrainedModel.REGISTRY)!r}"
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            backend = self.backend
            if (
                backend == "cuda"
                and torch.cuda.is_available()
                and callable(xlstm_available)
                and not xlstm_available()
            ):
                backend = "torch"
            return model_cls.from_pretrained(
                ckpt_path,
                backend=backend,
                device=device,
                compile=False,
            )
        return load_model(path, backend=self.backend)

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
        self._model = self._load_tirex_model()
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
