from typing import Any, Dict, Optional

import numpy as np
import torch
from momentfm import MOMENTPipeline
from pydantic import BaseModel as PydanticBaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tempus_bench.models.base_model import BaseModel, validate_inputs


class MomentHyperparams(PydanticBaseModel):
    pass


class MomentDataset(Dataset):
    """Dataset class for MOMENT model training and inference."""

    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int):
        self.data = data
        self.model_config["context_length"] = context_length
        self.prediction_length = prediction_length
        self.n_series, self.n_timesteps = data.shape
        self._scaler = StandardScaler()
        self._scaler.fit(data.T)
        self.scaled_data = self._scaler.transform(data.T).T

    def __len__(self):
        samples_per_series = max(
            0,
            self.n_timesteps
            - self.model_config["context_length"]
            - self.prediction_length
            + 1,
        )
        return self.n_series * samples_per_series

    def __getitem__(self, idx):
        samples_per_series = max(
            0,
            self.n_timesteps - self.context_length - self.prediction_length + 1,
        )
        series_idx = idx // samples_per_series
        time_idx = idx % samples_per_series
        start_idx = time_idx
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        context = self.scaled_data[series_idx, start_idx:context_end]
        target = self.scaled_data[series_idx, context_end:target_end]
        context = context.reshape(1, -1)
        target = target.reshape(1, -1)
        input_mask = np.ones(self.context_length)
        return (
            torch.FloatTensor(context),
            torch.FloatTensor(target),
            torch.FloatTensor(input_mask),
        )


class MomentModel(BaseModel):
    """MOMENT model wrapper for time series forecasting, extending FoundationModel."""

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, MomentHyperparams)
        self._scaler = StandardScaler()

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> "MomentModel":

        forecast_horizon = timestamps_target.shape[0]
        num_y_features = y_context.shape[1]

        self._model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": forecast_horizon,
                "n_channels": num_y_features,
                "head_dropout": self.head_dropout,
                "weight_decay": self.weight_decay,
                "freeze_encoder": self.freeze_encoder,
                "freeze_embedder": self.freeze_embedder,
                "freeze_head": self.freeze_head,
            },
        )
        self._model.init()
        self._model = self._model.to(self.device)

        self.is_fitted = True

        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Make predictions using the trained MOMENT model.

        Args:
            y_context: Recent/past target values
            timestamps_context: Timestamps for context data (not used)
            timestamps_target: Timestamps for target data (used for determining forecast_horizon)
            **kwargs: Additional arguments (unused)

        Returns:
            np.ndarray: Model predictions with shape (channels, forecast_horizon)
        """

        # Ensure y_context is 2D (num_steps, num_targets)
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
        _, num_targets = y_context.shape

        self._model.eval()

        with torch.no_grad():
            self._scaler.fit(y_context)
            y_context_scaled = self._scaler.transform(y_context)

            # Adjust input length to context_length
            if y_context_scaled.shape[0] >= self.context_window:
                y_context_trimmed = y_context_scaled[-self.context_window :, :]
            else:
                pad_rows = self.context_window - y_context_scaled.shape[0]
                y_context_trimmed = np.concatenate(
                    [np.zeros((pad_rows, num_targets)), y_context_scaled], axis=0
                )

            y_context_tensor = (
                torch.from_numpy(y_context_trimmed.T.copy())
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            input_mask = torch.ones(1, self.context_window, device=self.device)

            output = self._model(x_enc=y_context_tensor, input_mask=input_mask)
            # Output: (batch, forecast_horizon, channels)
            forecast = output.forecast.cpu().numpy()
            forecast = forecast[0, :, :].T  # (forecast_horizon, channels)
            forecast_scaled = self._scaler.inverse_transform(
                forecast
            )  # (channels, forecast_horizon)

        return forecast_scaled
