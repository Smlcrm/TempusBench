from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pydantic import BaseModel as PydanticBaseModel, Field
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from tempus_bench.models.base_model import BaseModel, validate_inputs


class MoiraiHyperparams(PydanticBaseModel):
    model_name: Optional[str] = Field(default="moirai", description="Model name")
    size: Optional[str] = Field(default=None, description="Model size")
    ctx: Optional[int] = Field(default=None, description="Context length")
    psz: Optional[int] = Field(default=16, ge=1, description="Patch size")
    bsz: Optional[int] = Field(default=32, ge=1, description="Batch size")
    test: Optional[int] = Field(default=100, ge=1, description="Test parameter")
    num_samples: Optional[int] = Field(default=100, ge=1, description="Number of samples")


class MoiraiModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Moirai model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, MoiraiHyperparams)

        self._model = None
        self.is_fitted = False

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "MoiraiModel":
        """
        "Train" the Moirai model (no training required for foundation models).

        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            timestamps_context: Timestamps for y_context (not used)
            timestamps_target: Timestamps for y_target (not used)
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance (for compatibility)
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]

        # Reference params, settings, device, python_version
        model_name = self.model_name
        size = self.size
        ctx = self.ctx
        psz = self.psz
        bsz = self.bsz
        test = self.test
        num_samples = self.num_samples

        # Prepare MoiraiForecast model with target_dim equal to num_targets

        if not self.is_fitted:
            pdt = y_target.shape[0]
            ctx = y_context.shape[0]
            self._model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(
                    pretrained_model_name_or_path=f"Salesforce/{model_name}-1.1-R-{size}"
                ),
                prediction_length=pdt,
                context_length=ctx,
                patch_size=psz,
                num_samples=num_samples,
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        self.is_fitted = True
        return self

    @validate_inputs
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

        ctx = self._model_config["ctx"]
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

        forecast = self._model(
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