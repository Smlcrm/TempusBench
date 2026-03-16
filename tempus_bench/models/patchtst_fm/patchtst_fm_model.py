"""
IBM PatchTST-FM pretrained foundation model for time series forecasting.

Uses the zero-shot pretrained PatchTST-FM (ibm-research/patchtst-fm-r1) from IBM,
a ~260M parameter model that achieves state-of-the-art results on GIFT-Eval.
Requires granite-tsfm from the patchtst-fm branch.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support

try:
    from tsfm_public import PatchTSTFMForPrediction
except ImportError as e:
    PatchTSTFMForPrediction = None
    _PATCHTST_FM_IMPORT_ERROR = str(e)


class PatchtstFmHyperparams(PydanticBaseModel):
    pass


class PatchtstFmModel(BaseModel):
    """
    IBM PatchTST-FM pretrained foundation model wrapper.

    Zero-shot stochastic forecasting via the granite-tsfm package (patchtst-fm branch).
    PatchTST-FM outputs quantile forecasts which are converted to pseudo-samples.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, PatchtstFmHyperparams)
        if PatchTSTFMForPrediction is None:
            raise ImportError(
                "PatchTST-FM requires granite-tsfm from patchtst-fm branch. "
                "Install with: pip install 'git+https://github.com/ibm-granite/granite-tsfm.git@patchtst-fm'. "
                f"Original error: {_PATCHTST_FM_IMPORT_ERROR}"
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
    ) -> "PatchtstFmModel":
        """Load pretrained PatchTST-FM (no training required for foundation models)."""
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="PatchTST-FM",
        )
        device = getattr(self, "device", "cpu")
        self._model = PatchTSTFMForPrediction.from_pretrained(self.hf_model_name)
        self._model = self._model.to(device)
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
            raise ValueError("PatchtstFmModel is not fitted. Call train() first.")

        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="PatchTST-FM",
        )
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        num_samples = kwargs.get("num_samples", 100)

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        # PatchTST-FM expects list of 1D tensors (one per channel, channel-independent)
        n_channels = y_input.shape[1]
        device = getattr(self, "device", "cpu")
        inputs_list: List[torch.Tensor] = [
            torch.tensor(y_input[:, i], dtype=torch.float32, device=device)
            for i in range(n_channels)
        ]

        with torch.no_grad():
            output = self._model(
                inputs_list,
                prediction_length=forecast_horizon,
                return_loss=False,
            )

        # output.quantile_predictions: (n_channels, quantiles, forecast_len)
        quantile_preds = output.quantile_predictions
        pred = quantile_preds.cpu().numpy()
        # Keep first num_targets channels; transpose to (quantiles, forecast_len, num_targets)
        if pred.ndim == 3:
            pred = pred[:num_targets, :, :]
        if pred.ndim == 3 and pred.shape[0] == num_targets:
            pred = np.transpose(pred, (1, 2, 0))
        elif pred.ndim == 2:
            pred = pred[:, :, np.newaxis]

        n_quantiles = pred.shape[0]
        pred = pred[:, :forecast_horizon, :]  # (quantiles, forecast_horizon, num_targets)

        # Convert quantiles to pseudo-samples
        indices = np.linspace(0, n_quantiles - 1, num_samples, dtype=int)
        samples = pred[indices, :, :]  # (num_samples, forecast_horizon, num_targets)

        return samples.astype(np.float64)
