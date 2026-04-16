"""
Chronos-Bolt foundation model implementation for time series forecasting.

Chronos-Bolt is a more efficient variant of Chronos that uses a T5-based
encoder-decoder architecture optimized for speed. It is loaded via
BaseChronosPipeline.from_pretrained which auto-detects the Bolt variant.

Requires chronos-forecasting>=2.0.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from chronos import BaseChronosPipeline
except ImportError as e:
    BaseChronosPipeline = None
    _CHRONOS_BOLT_IMPORT_ERROR = str(e)


class ChronosBoltBaseHyperparams(PydanticBaseModel):
    pass


class ChronosBoltBaseModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronosBoltBaseHyperparams)
        if BaseChronosPipeline is None:
            raise ImportError(
                "Chronos-Bolt requires chronos-forecasting>=2.0. "
                f"Install with: pip install 'chronos-forecasting>=2.0'. "
                f"Original error: {_CHRONOS_BOLT_IMPORT_ERROR}"
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
    ) -> "ChronosBoltBaseModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Chronos-Bolt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = "auto" if device == "cuda" else "cpu"

        self._model = BaseChronosPipeline.from_pretrained(
            self.hf_model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
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
            supports_both=False,
            model_name="Chronos-Bolt",
        )
        context_length = min(self.context_length, y_context.shape[0])
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        if y_input.shape[0] >= context_length:
            y_trimmed = y_input[-context_length:, :]
        else:
            pad_rows = context_length - y_input.shape[0]
            y_trimmed = np.pad(y_input, ((pad_rows, 0), (0, 0)), mode="edge")

        inputs = torch.tensor(y_trimmed.T, dtype=torch.float32)
        forecasts = self._model.predict(
            inputs=inputs, prediction_length=forecast_horizon
        )
        forecasts = np.asarray(forecasts)

        # Keep only first num_targets channels; transpose to (num_samples, forecast_horizon, num_targets)
        forecasts = forecasts[:num_targets, :, :]
        forecasts = np.transpose(forecasts, (1, 2, 0))
        return forecasts
