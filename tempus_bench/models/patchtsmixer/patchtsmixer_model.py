"""
PatchTSMixer model for multivariate time series forecasting.

PatchTSMixer is a lightweight MLP-Mixer architecture that mixes across
patches, channels, and hidden features — providing native multivariate
support with cross-channel correlation learning.

Output: deterministic (point forecasts via MSE loss).
Multivariate: native channel-mixing.
Covariates: not supported.
Autoregressive: no (direct multi-step).

Trained from scratch on context data via sliding-window optimization
(same pattern as PatchTST in TempusBench).

Paper: "TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series
        Forecasting" (arXiv:2306.09364, KDD 2023)
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs

try:
    from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
except ImportError as e:
    PatchTSMixerConfig = None
    PatchTSMixerForPrediction = None
    _PATCHTSMIXER_IMPORT_ERROR = str(e)


class PatchtsmixerHyperparams(PydanticBaseModel):
    pass


class PatchtsmixerModel(BaseModel):
    """
    PatchTSMixer model with native multivariate channel-mixing.

    Initialized from a configuration (not a pretrained checkpoint) and
    briefly fit on the provided context data via sliding-window training.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, PatchtsmixerHyperparams)
        if PatchTSMixerForPrediction is None:
            raise ImportError(
                "PatchTSMixer requires transformers>=4.36.0. "
                f"Install with: pip install 'transformers>=4.36.0'. "
                f"Original error: {_PATCHTSMIXER_IMPORT_ERROR}"
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
    ) -> "PatchtsmixerModel":
        ctx_len = min(self.context_length, y_context.shape[0])
        pred_len = y_target.shape[0]
        num_channels = y_context.shape[1]

        config = PatchTSMixerConfig(
            context_length=ctx_len,
            prediction_length=pred_len,
            num_input_channels=num_channels,
            patch_length=self.patch_length,
            stride=self.stride,
        )
        self._model = PatchTSMixerForPrediction(config)

        self._train_on_context(y_context, ctx_len, pred_len)
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
            raise ValueError("PatchtsmixerModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        ctx_len = self._model.config.context_length

        context = y_context[-ctx_len:, :]
        if context.shape[0] < ctx_len:
            pad_rows = ctx_len - context.shape[0]
            context = np.pad(context, ((pad_rows, 0), (0, 0)), mode="edge")

        # PatchTSMixer (HF) expects (batch, sequence_length, num_input_channels)
        # Pipeline convention: y_context (num_steps, num_targets) -> (1, seq_len, channels)
        input_tensor = torch.tensor(
            context[np.newaxis, :, :], dtype=torch.float32
        )

        self._model.eval()
        with torch.no_grad():
            output = self._model(past_values=input_tensor)

        # output.prediction_outputs: (batch, prediction_length, num_input_channels)
        preds = output.prediction_outputs[0].numpy()  # (prediction_length, num_channels)
        return preds[:forecast_horizon].astype(np.float64)

    def _train_on_context(
        self, y_context: np.ndarray, ctx_len: int, pred_len: int
    ):
        window_size = ctx_len + pred_len
        n_steps = y_context.shape[0]

        # HF PatchTSMixer expects (batch, sequence_length, num_input_channels)
        # Pipeline convention: y_context (num_steps, num_targets)
        if n_steps < window_size:
            x_windows = y_context[:ctx_len, :][np.newaxis, :, :]
            y_windows = y_context[ctx_len : ctx_len + pred_len, :][np.newaxis, :, :]
            if y_windows.shape[1] == 0:
                return
        else:
            starts = list(range(0, n_steps - window_size + 1))
            x_windows = np.stack(
                [y_context[s : s + ctx_len, :] for s in starts], axis=0
            )
            y_windows = np.stack(
                [y_context[s + ctx_len : s + window_size, :] for s in starts], axis=0
            )

        x_tensor = torch.tensor(x_windows, dtype=torch.float32)
        y_tensor = torch.tensor(y_windows, dtype=torch.float32)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )

        self._model.train()
        for _ in range(self.num_epochs):
            output = self._model(
                past_values=x_tensor, future_values=y_tensor
            )
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
