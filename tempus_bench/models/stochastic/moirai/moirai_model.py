import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
from typing import Optional, List, Union
from einops import rearrange
from tempus_bench.models.base_model import BaseModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class MoiraiModel(BaseModel):

    def __init__(self, config: UnifiedConfig, logs_path: str):
        """
        Args:
          config: Configuration dictionary containing model parameters
          logs_path: Directory for storing log files (optional)
        """

        super().__init__(config_path, logs_path, hyperparameters)

        # Set reasonable defaults for all model-specific parameters if not provided in config
        # As in https://arxiv.org/pdf/2402.02592
        self.model_config["model_name"] = "moirai"
        self.model_config["size"] = self.model_config.get("size")
        self.model_config["ctx"] = None
        self.model_config["psz"] = 16
        self.model_config["bsz"] = 32
        self.model_config["test"] = 100
        self.model_config["num_samples"] = 100

        self.model = None
        self.is_fitted = False

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "MoiraiModel":
        """
        "Train" the Moirai model (no training required for foundation models).

        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            timestamps_context: Timestamps for y_context (not used)
            timestamps_target: Timestamps for y_target (not used)
            freq: Frequency string (required by interface, not used)

        Returns:
            self: The fitted model instance (for compatibility)
        """
        # Prepare MoiraiForecast model with target_dim equal to num_targets

        if not self.is_fitted:
            self.model_config["pdt"] = y_target.shape[0]
            self.model_config["ctx"] = y_context.shape[0]
            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(
                    pretrained_model_name_or_path=f"Salesforce/{self.model_config['model_name']}-1.1-R-{self.model_config['size']}"
                ),
                prediction_length=self.model_config["pdt"],
                context_length=self.model_config["ctx"],
                patch_size=self.model_config["psz"],
                num_samples=self.model_config["num_samples"],
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> np.ndarray:
        """
        Make predictions using the Moirai model.

        Args:
            y_context: Recent/past target values, shape (context_steps, num_targets)
            timestamps_context: Timestamps for y_context (not used for prediction)
            timestamps_target: Timestamps for the prediction horizon (used to determine forecast length)
            freq: Frequency string (must be provided from CSV data, required)

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)

        Raises:
            ValueError: If model is not fitted, freq is not provided, or forecast length cannot be determined
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        prediction_length = timestamps_target.shape[0]

        context_steps, num_targets = y_context.shape

        ctx = self.model_config["ctx"]
        # Create mask with the padded size (ctx, num_targets)
        observed_mask = np.ones((ctx, num_targets), dtype=bool)

        # Prepare past_target tensor: shape (1, ctx, num_targets)
        past_target = torch.tensor(y_context, dtype=torch.float32).unsqueeze(0)

        # past_observed_target: True where value is observed, False where padded (1, ctx, num_targets)
        past_observed_target = torch.tensor(observed_mask, dtype=torch.bool).unsqueeze(
            0
        )
        # past_is_pad: True where ANY variate at a timestep is padded, False otherwise (1, ctx)
        past_is_pad = (
            (~torch.tensor(observed_mask, dtype=torch.bool)).any(dim=-1).unsqueeze(0)
        )

        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        # forecast shape: (num_targets, num_samples, prediction_length)
        # Convert to numpy array
        forecast_np = (
            forecast.cpu().numpy() if hasattr(forecast, "cpu") else np.array(forecast)
        )

        # Transpose from (num_targets, num_samples, prediction_length) to (num_samples, prediction_length, num_targets)
        # Then the base class will handle point forecasts if needed
        samples = np.transpose(forecast_np, (1, 2, 0))
        
        # If univariate, ensure shape is (num_samples, prediction_length, 1)
        if samples.ndim == 2:
            samples = samples[:, :, np.newaxis]
        
        return samples