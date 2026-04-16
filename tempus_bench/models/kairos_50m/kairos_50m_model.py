"""
Kairos foundation model for time series forecasting.

Kairos is a parameter-efficient TSFM from ShanghaiTech/Ant Group featuring:
- Mixture-of-Size Encoder for adaptive patching
- Heterogeneity-Aware Transformer with Dynamic Rotary Position Embedding
- Multi-Patch Decoder for parallel multi-step prediction

Output: 9 quantiles (0.1–0.9) — stochastic.
Multivariate: channel-independent (iterates over targets).
Covariates: not supported.
Autoregressive: direct within native horizon (64 steps); autoregressive beyond.

Paper: "Kairos: Towards Adaptive and Generalizable Time Series Foundation Models"
       arXiv:2509.25826 (ICML 2025 submission)
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tsfm.model.kairos import AutoModel as KairosAutoModel
except ImportError as e:
    KairosAutoModel = None
    _KAIROS_IMPORT_ERROR = str(e)


class Kairos50mHyperparams(PydanticBaseModel):
    pass


class Kairos50mModel(BaseModel):
    """
    Kairos foundation model wrapper with stochastic (quantile) output.

    Returns 9 quantile forecasts (0.1–0.9) converted to pseudo-samples
    for stochastic metrics. Processes each target independently.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, Kairos50mHyperparams)
        if KairosAutoModel is None:
            raise ImportError(
                "Kairos requires the kairos package. "
                "Install with: pip install git+https://github.com/foundation-model-research/Kairos. "
                f"Original error: {_KAIROS_IMPORT_ERROR}"
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
    ) -> "Kairos50mModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Kairos",
        )
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
        if not self.is_fitted:
            raise ValueError("Kairos50mModel is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Kairos",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 100)

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        context_length = min(self.context_length, y_input.shape[0])
        if y_input.shape[0] >= context_length:
            y_trimmed = y_input[-context_length:, :]
        else:
            pad_rows = context_length - y_input.shape[0]
            y_trimmed = np.pad(y_input, ((pad_rows, 0), (0, 0)), mode="edge")

        n_channels = y_trimmed.shape[1]
        device = next(self._model.parameters()).device
        all_quantiles = []

        for i in range(n_channels):
            series = torch.tensor(
                y_trimmed[:, i].astype(np.float32), device=device
            ).unsqueeze(0)  # (1, context_length)

            with torch.no_grad():
                output = self._model(
                    past_target=series,
                    prediction_length=forecast_horizon,
                    generation=True,
                    infer_is_positive=True,
                    force_flip_invariance=True,
                )

            # prediction_outputs: (1, n_quantiles, prediction_length)
            preds = output["prediction_outputs"][0].cpu().numpy()
            # preds: (n_quantiles, prediction_length)
            all_quantiles.append(preds)

        # all_quantiles: list of (n_quantiles, forecast_horizon), one per channel
        # Stack and keep only first num_targets (discard covariate channel forecasts)
        quantiles = np.stack(all_quantiles, axis=0)[:num_targets, :, :]
        n_quantiles = quantiles.shape[1]

        # Convert quantiles to pseudo-samples
        indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
        samples = quantiles[:, indices, :]  # (num_targets, num_samples, forecast_horizon)

        # Target shape: (num_samples, forecast_horizon, num_targets)
        samples = np.transpose(samples, (1, 2, 0))
        return samples.astype(np.float64)

    def _build_model(self):
        self._model = KairosAutoModel.from_pretrained(
            self.hf_model_name, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()
