import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel
from momentfm import MOMENTPipeline
from tqdm import tqdm


class MomentDataset(Dataset):
    """Dataset class for MOMENT model training and inference."""

    def __init__(
        self,
        data: np.ndarray,
        context_length: int,
        prediction_length: int,
        scaler: Optional[StandardScaler] = None,
    ):
        self.data = data
        self.model_config["context_length"] = context_length
        self.prediction_length = prediction_length
        self.n_series, self.n_timesteps = data.shape
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data.T)
        else:
            self.scaler = scaler
        self.scaled_data = self.scaler.transform(data.T).T

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
            self.n_timesteps
            - self.model_config["context_length"]
            - self.prediction_length
            + 1,
        )
        series_idx = idx // samples_per_series
        time_idx = idx % samples_per_series
        start_idx = time_idx
        context_end = start_idx + self.model_config["context_length"]
        target_end = context_end + self.prediction_length
        context = self.scaled_data[series_idx, start_idx:context_end]
        target = self.scaled_data[series_idx, context_end:target_end]
        context = context.reshape(1, -1)
        target = target.reshape(1, -1)
        input_mask = np.ones(self.model_config["context_length"])
        return (
            torch.FloatTensor(context),
            torch.FloatTensor(target),
            torch.FloatTensor(input_mask),
        )


class MomentModel(BaseModel):
    """MOMENT model wrapper for time series forecasting, extending FoundationModel."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            config,
        )
        self.model_config["context_length"] = 512

        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"MOMENT Model initialized - Device: {self.device}")
        print(f"Context length: {self.model_config['context_length']}")

    def _load_model(self, forecast_horizon: int):
        print(f"Loading MOMENT model for forecast horizon: {forecast_horizon}")
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": forecast_horizon,
                "n_channels": self.model_config["num_y_features"],
                "head_dropout": 0.1,
                "weight_decay": 0,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            },
        )
        self.model.init()
        self.model = self.model.to(self.device)
        print("MOMENT model loaded successfully!")

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "MomentModel":
        """
        Train/fine-tune the MOMENT model on given data.

        Args:
            y_context: Past target values
            y_target: Future target values (not used for MOMENT)
            y_start_date: Start date timestamp (not used for MOMENT)

        Returns:
            self: The fitted model instance
        """

        return self

    def predict(
        self,
        y_context:np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained MOMENT model.

        Args:
            y_context: Recent/past target values
            y_target: Future target values (not used for MOMENT)
            y_context_timestamps: Timestamps for context data (not used for MOMENT)
            y_target_timestamps: Timestamps for target data (not used for MOMENT)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            **kwargs: Additional arguments (ignored)

        Returns:
            np.ndarray: Model predictions with shape (forecast_horizon,)
        """

        forecast_horizon = timestamps_target.shape[0]
        num_y_features = y_context.shape[1]
        self.model_config["num_y_features"] = num_y_features

        self._load_model(forecast_horizon)

        self.model.eval()

        with torch.no_grad():

            # Scale the context data using the fitted scaler
            self.scaler.fit(y_context)
            y_context = self.scaler.transform(y_context)

            if y_context.shape[0] >= self.model_config["context_length"]:
                y_context = y_context[-self.model_config["context_length"] :, :]
            else:
                padding = np.zeros(
                    self.model_config["context_length"] - len(y_context)
                )
                padding = np.expand_dims(padding, axis=1)
                y_context = np.concatenate([padding, y_context], axis=0).T
                warnings.warn(
                    f"Time Series is shorter than context_length {self.model_config['context_length']}. "
                    "Padded with zeros.",
                    UserWarning,
                )

            # Create proper 3D tensor: [batch_size=1, sequence_length=1, features=1]
            y_context = torch.FloatTensor(y_context.T).unsqueeze(0).to(self.device)

            # Create input mask
            input_mask = torch.ones(1, self.model_config["context_length"]).to(
                self.device
            )

            # Debug shapes and contents before passing to model
            print(f"y_context shape: {y_context.shape}, dtype: {y_context.dtype}")
            print(f"input_mask shape: {input_mask.shape}, dtype: {input_mask.dtype}")
            print(f"y_context (sample): {y_context[0, :5, :].cpu().numpy() if y_context.shape[1] >= 5 else y_context[0, :, :].cpu().numpy()}")
            print(f"input_mask (sample): {input_mask[0, :5].cpu().numpy() if input_mask.shape[1] >= 5 else input_mask[0, :].cpu().numpy()}")
            output = self.model(x_enc=y_context, input_mask=input_mask)

            # Inverse scale the forecast
            forecast_scaled = output.forecast.cpu().numpy()
            forecast = self.scaler.inverse_transform(forecast_scaled[0, :, :].T)
            # forecast = forecast_scaled[0, :, :].T

        return forecast
