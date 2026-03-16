"""
Chronos-2 foundation model implementation for time series forecasting with covariate support.

This module provides a wrapper around Amazon Chronos-2, which natively supports:
- Past-only covariates (real/categorical)
- Known future covariates (real/categorical)
- Univariate and multivariate forecasting

Requires chronos-forecasting>=2.0.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs

try:
    from chronos import Chronos2Pipeline
except ImportError as e:
    Chronos2Pipeline = None
    _CHRONOS2_IMPORT_ERROR = str(e)


class Chronos2Hyperparams(PydanticBaseModel):
    pass


class Chronos2Model(BaseModel):
    """
    Chronos-2 foundation model wrapper with native covariate support.

    Chronos-2 supports past-only and known future covariates. When covariates
    are provided, they are passed to the model. When not provided, forecasting
    uses only the target series (same as Chronos).
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, Chronos2Hyperparams)
        if Chronos2Pipeline is None:
            raise ImportError(
                "Chronos-2 requires chronos-forecasting>=2.0. "
                f"Install with: pip install 'chronos-forecasting>=2.0'. Original error: {_CHRONOS2_IMPORT_ERROR}"
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
    ) -> "Chronos2Model":
        """
        Initialize Chronos-2 (no training required for foundation models).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = "auto" if device == "cuda" else None

        self._model = Chronos2Pipeline.from_pretrained(
            self.hf_model_name,
            device_map=device_map or "cpu",
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
        """
        Make predictions using Chronos-2. Uses covariates when provided.

        Chronos-2 API (list of dicts):
        - target: (n_variates, history_length)
        - past_covariates: dict of 1D arrays, length = history_length
        - future_covariates: dict of 1D arrays, length = prediction_length;
          keys must be subset of past_covariates
        """
        context_length = min(self.context_length, y_context.shape[0])
        forecast_horizon = timestamps_target.shape[0]
        num_samples = kwargs.get("num_samples", 100)
        use_covariates = kwargs.get("use_covariates", True)

        # Trim or pad context
        if y_context.shape[0] >= context_length:
            y_trimmed = y_context[-context_length:, :]
            if x_context is not None and use_covariates:
                x_context_trimmed = x_context[-context_length:, :]
            else:
                x_context_trimmed = None
        else:
            pad_rows = context_length - y_context.shape[0]
            y_trimmed = np.pad(y_context, ((pad_rows, 0), (0, 0)), mode="edge")
            if x_context is not None and use_covariates:
                x_context_trimmed = np.pad(x_context, ((pad_rows, 0), (0, 0)), mode="edge")
            else:
                x_context_trimmed = None

        # Chronos2 expects (n_variates, history_length)
        target = np.asarray(y_trimmed.T, dtype=np.float32)

        # Build input dict for Chronos2 predict()
        # Chronos2 supports past-only, future-only, or both (optional, independent)
        input_dict = {"target": target}
        if x_context_trimmed is not None:
            input_dict["past_covariates"] = {
                f"cov_{i}": x_context_trimmed[:, i].astype(np.float32)
                for i in range(x_context_trimmed.shape[1])
            }
        if x_target is not None and use_covariates:
            input_dict["future_covariates"] = {
                f"cov_{i}": x_target[:forecast_horizon, i].astype(np.float32)
                for i in range(x_target.shape[1])
            }

        # Chronos2 predict returns list of (n_variates, n_quantiles, prediction_length)
        predictions = self._model.predict(
            [input_dict],
            prediction_length=forecast_horizon,
            batch_size=1,
        )

        # predictions: list of 1 element, shape (n_variates, n_quantiles, prediction_length)
        pred = predictions[0]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # We have quantiles; use them as pseudo-samples for stochastic metrics
        # pred shape: (n_variates, n_quantiles, prediction_length)
        # Target: (num_samples, forecast_horizon, num_targets)
        n_variates, n_quantiles, _ = pred.shape

        if n_quantiles >= num_samples:
            # Use first num_samples quantiles as samples
            samples = pred[:, :num_samples, :]  # (n_variates, num_samples, horizon)
        else:
            # Repeat quantiles to reach num_samples (interpolate or repeat)
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = pred[:, indices, :]  # (n_variates, num_samples, horizon)

        # Transpose to (num_samples, forecast_horizon, num_targets)
        samples = np.transpose(samples, (1, 2, 0))
        return samples.astype(np.float64)
