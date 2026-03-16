from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tsfm_public import FlowStateForPrediction
except ImportError as e:
    FlowStateForPrediction = None
    _GRANITE_IMPORT_ERROR = str(e)

FREQ_TO_SCALE_FACTOR = {
    "15min": 0.0104,
    "15T": 0.0104,
    "30min": 0.0208,
    "30T": 0.0208,
    "h": 0.0417,
    "H": 0.0417,
    "1H": 0.0417,
    "D": 1.0,
    "1D": 1.0,
    "W": 7.0,
    "1W": 7.0,
    "M": 30.44,
    "1M": 30.44,
    "MS": 30.44,
}


class GraniteFlowstateHyperparams(PydanticBaseModel):
    pass


class GraniteFlowstateModel(BaseModel):
    """
    IBM Granite FlowState foundation model wrapper.

    Zero-shot stochastic forecasting via the granite-tsfm package.
    FlowState outputs 9 quantile forecasts (shape: batch, quantiles, horizon, channels)
    which are converted to pseudo-samples for stochastic metrics.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, GraniteFlowstateHyperparams)
        if FlowStateForPrediction is None:
            raise ImportError(
                "Granite FlowState requires granite-tsfm>=0.3.0. "
                f"Install with: pip install 'granite-tsfm>=0.3.0'. Original error: {_GRANITE_IMPORT_ERROR}"
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
    ) -> "GraniteFlowstateModel":
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Granite FlowState",
        )
        self._model = FlowStateForPrediction.from_pretrained(
            self.hf_model_name
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
        if not self.is_fitted:
            raise ValueError(
                "GraniteFlowstateModel is not fitted. Call train() first."
            )

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Granite FlowState",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1] if y_context.ndim == 2 else 1
        num_samples = kwargs.get("num_samples", 100)
        freq = kwargs.get("freq", None)
        scale_factor = FREQ_TO_SCALE_FACTOR.get(freq, 1.0) if freq else 1.0

        # FlowState supports only single variate per call. Use Lag-Llama-style iteration:
        # iterate over targets + covariates (one univariate call per variate), discard covariate predictions.
        if x_context is not None:
            variates = np.concatenate([y_context, x_context], axis=1)
        else:
            variates = y_context if y_context.ndim == 2 else y_context[:, np.newaxis]

        all_samples = []
        for k in range(variates.shape[1]):
            y_single = variates[:, k : k + 1]  # (context_len, 1)

            # FlowState expects (context_length, batch_size, n_channels) with batch_first=False
            # Must pass only the target channel - no covariates (raises if >1 channel)
            context_tensor = torch.tensor(
                y_single[:, np.newaxis, :], dtype=torch.float32
            )

            with torch.no_grad():
                forecast = self._model(
                    context_tensor,
                    scale_factor=scale_factor,
                    prediction_length=forecast_horizon,
                    batch_first=False,
                )

            # Use quantile_outputs when available (for pseudo-samples); else prediction_outputs (mean)
            out = forecast
            if hasattr(out, "quantile_outputs") and out.quantile_outputs is not None:
                pred_single = out.quantile_outputs
            else:
                pred_single = out.prediction_outputs
            if isinstance(pred_single, torch.Tensor):
                pred_single = pred_single.cpu().numpy()
            # quantile_outputs: 5D (num_ch, batch, n_quants, horizon, 1); prediction_outputs: 3D (batch, horizon, 1)
            if pred_single.ndim == 5:
                pred_single = pred_single[0, 0, :, :, 0][:, :, np.newaxis]  # (n_quantiles, horizon, 1)
            elif pred_single.ndim == 4:
                pred_single = pred_single[0, :, :, :1]  # (n_quantiles, horizon, 1)
            else:
                pred_single = pred_single[0, :, np.newaxis][np.newaxis, :, :]  # (1, horizon, 1)
            all_samples.append(pred_single)

        # Use only first num_targets outputs (target predictions); discard covariate predictions
        all_samples = all_samples[:num_targets]

        # Stack: (n_quantiles, horizon, num_targets) or (1, horizon, num_targets) for mean mode
        pred = np.concatenate(all_samples, axis=-1)
        n_quantiles = pred.shape[0]

        # Truncate to forecast_horizon
        pred = pred[:, :forecast_horizon, :]

        # Convert quantiles to pseudo-samples (or replicate mean if single point)
        if n_quantiles > 1:
            indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
            samples = pred[indices, :, :]  # (num_samples, forecast_horizon, num_targets)
        else:
            samples = np.tile(pred, (num_samples, 1, 1))

        return samples.astype(np.float64)
