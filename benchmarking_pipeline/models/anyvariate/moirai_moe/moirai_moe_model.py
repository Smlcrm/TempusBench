import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
from typing import Optional, List, Union
from einops import rearrange
from benchmarking_pipeline.models.base_model import BaseModel
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


class MoiraiMoeModel(BaseModel):

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
          model_name: the type of moirai model you want to use - choose from {'moirai', 'moirai_moe'}
          size: the model size - choose from {'small', 'base', 'large'}
          pdt: prediction length - any positive integer
          ctz: context length - any positive integer
          psz: patch size - choose from {"auto", 8, 16, 32, 64, 128}
          bsz: batch size - any positive integer
          test: test set length - any positive integer
          num_samples: number of samples to generate during prediction time - any positive integer
        """

        super().__init__(config)

        # Set reasonable defaults for all model-specific parameters if not provided in config
        # As in https://arxiv.org/pdf/2402.02592
        self.model_config["model_name"] = "moirai-moe"
        self.model_config["size"] = self.model_config.get("size", "small")
        self.model_config["ctx"] = None
        self.model_config["psz"] = "auto"
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
    ) -> "MoiraiMoeModel":
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
            print(f"[DEBUG] pdt: {self.model_config['pdt']}")
            self.model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(
                    pretrained_model_name_or_path = f"Salesforce/{self.model_config['model_name']}-1.0-R-{self.model_config['size']}"
                ),
                prediction_length=self.model_config["pdt"],
                context_length=self.model_config["ctx"],
                patch_size=self.model_config["psz"],
                num_samples=self.model_config["num_samples"],
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0
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
            np.ndarray: Model predictions with shape (prediction_length, num_targets)

        Raises:
            ValueError: If model is not fitted, freq is not provided, or forecast length cannot be determined
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        prediction_length = timestamps_target.shape[0]
        # y_context is always (context_steps, num_targets)

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

        # Debug: Print tensor shapes
        print(f"[DEBUG] past_target shape: {past_target.shape}")
        print(f"[DEBUG] past_observed_target shape: {past_observed_target.shape}")
        print(f"[DEBUG] past_is_pad shape: {past_is_pad.shape}")
        print(f"[DEBUG] ctx: {ctx}, num_targets: {num_targets}")
        print(f"[DEBUG] y_context original shape: {y_context.shape}")
        print(f"[DEBUG] y_context_padded shape: {y_context.shape}")

        # Debug: Print actual values
        print(f"[DEBUG] past_target non-zero values: {(past_target != 0).sum()}")
        print(f"[DEBUG] past_observed_target True values: {past_observed_target.sum()}")
        print(f"[DEBUG] past_is_pad True values: {past_is_pad.sum()}")

        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        # forecast[0] shape: (num_samples, prediction_length, num_targets)
        # Ensure forecast[0] is a numpy array before taking mean
        forecast_np = (
            forecast[0].cpu().numpy()
            if hasattr(forecast[0], "cpu")
            else np.array(forecast[0])
        )

        # The error is: IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        # This is because forecasted_values is 1D, but the code tries to index it as 2D: forecasted_values[:prediction_length, :]

        forecasted_values = np.round(np.mean(forecast_np, axis=0), decimals=4)

        pdt = self.model_config["pdt"]

        print(f"[DEBUG] forecasted_values shape before any reshape: {forecasted_values.shape}")
        print(f"[DEBUG] prediction_length: {prediction_length}")
        print(f"[DEBUG] pdt from model_config: {pdt}")
        print(f"[DEBUG] num_targets: {num_targets}")
        print(f"[DEBUG] forecasted_values.size: {forecasted_values.size}")

        # If forecasted_values is 1D, try to reshape to (pdt, num_targets) if possible
        if forecasted_values.ndim == 1:
            print(f"[DEBUG] forecasted_values is 1D, attempting to reshape to ({pdt}, {num_targets})")
            if forecasted_values.size == pdt * num_targets:
                forecasted_values = forecasted_values.reshape((pdt, num_targets))
                print(f"[DEBUG] Reshaped forecasted_values to {forecasted_values.shape}")
            else:
                print(f"[DEBUG] Cannot reshape: forecasted_values.size={forecasted_values.size}, expected={pdt * num_targets}")
                # Fallback: expand dims to (pdt, 1) if num_targets==1 and forecasted_values.size==pdt
                if num_targets == 1 and forecasted_values.size == pdt:
                    forecasted_values = forecasted_values.reshape((pdt, 1))
                    print(f"[DEBUG] Reshaped forecasted_values to {forecasted_values.shape} (single target)")
                else:
                    raise ValueError(
                        f"forecasted_values shape {forecasted_values.shape} is incompatible with expected ({pdt}, {num_targets})"
                    )

        if forecasted_values.ndim == 2:
            print(f"[DEBUG] forecasted_values is 2D, shape: {forecasted_values.shape}")
            if forecasted_values.shape[0] != pdt or forecasted_values.shape[1] != num_targets:
                print(f"[DEBUG] forecasted_values shape mismatch: {forecasted_values.shape} (expected: ({pdt}, {num_targets}))")

        if forecasted_values.shape[0] < prediction_length:
            print(f"[DEBUG] Model returned fewer forecast steps ({forecasted_values.shape[0]}) than requested ({prediction_length}).")

        forecast_matrix = forecasted_values[:prediction_length, :]
        print(forecast_matrix)
        print(forecast_matrix.shape)

        # self._last_y_pred = forecast_matrix
        return forecast_matrix
