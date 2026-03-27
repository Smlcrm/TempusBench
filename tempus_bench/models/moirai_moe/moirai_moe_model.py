from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel, Field
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from tempus_bench.models.base_model import BaseModel, validate_inputs


class MoiraiMoeHyperparams(PydanticBaseModel):
    pass


class MoiraiMoeModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Moirai model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, MoiraiMoeHyperparams)

        self._model = None
        self.is_fitted = False

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
            train_span = int(y_target.shape[0])
            val_h_raw = kwargs.get("validate_horizon")
            val_h = int(val_h_raw) if val_h_raw is not None else train_span
            pdt = max(train_span, val_h)
            ctx = y_context.shape[0]
            feat_dim = x_target.shape[1] if x_target is not None else 0
            past_feat_dim = x_context.shape[1] if x_context is not None else 0
            self._model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(
                    pretrained_model_name_or_path=self.hf_model_name
                ),
                prediction_length=pdt,
                context_length=ctx,
                patch_size=pdt + ctx,
                num_samples=kwargs["num_samples"],
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=feat_dim,
                past_feat_dynamic_real_dim=past_feat_dim,
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
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Make predictions using the MoiraiMoE model.

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
        num_targets = y_context.shape[1]

        ctx = y_context.shape[0]
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

        past_feat_dynamic_real = None
        past_observed_feat_dynamic_real = None
        feat_dynamic_real = None
        observed_feat_dynamic_real = None
        forecast_horizon = timestamps_target.shape[0]
        # Moirai-MoE: past-only, both, or future-only. Future-only zero-pads past.
        # observed_feat_dynamic_real must be provided if feat_dynamic_real is provided (uni2ts API).
        if x_context is not None:
            past_feat_dynamic_real = torch.tensor(
                x_context, dtype=torch.float32
            ).unsqueeze(0)
            past_observed_feat_dynamic_real = torch.ones_like(
                past_feat_dynamic_real, dtype=torch.bool
            )
        if x_target is not None:
            x_future = x_target[:forecast_horizon]
            if x_context is not None:
                feat_arr = np.concatenate([x_context, x_future], axis=0)
            else:
                ctx = y_context.shape[0]
                feat_arr = np.concatenate(
                    [np.zeros((ctx, x_future.shape[1]), dtype=np.float32), x_future],
                    axis=0,
                )
            feat_dynamic_real = torch.tensor(
                feat_arr, dtype=torch.float32
            ).unsqueeze(0)
            observed_feat_dynamic_real = torch.ones_like(
                feat_dynamic_real, dtype=torch.bool
            )

        forecast = self._model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
        )

        # uni2ts MoiraiMoE forward outputs (batch, num_targets, num_samples, prediction_length)
        # or (num_targets, num_samples, prediction_length) after dropping batch.
        # MetricRegistry stochastic path expects (num_samples, prediction_length, num_targets).
        forecast = np.asarray(forecast)
        if forecast.ndim >= 1 and forecast.shape[0] == 1:
            forecast = np.squeeze(forecast, axis=0)

        forecast_horizon = int(timestamps_target.shape[0])
        if forecast.ndim == 3 and forecast.shape[0] == num_targets:
            forecast = np.transpose(forecast, (1, 2, 0))
        elif forecast.ndim == 2:
            forecast = np.expand_dims(forecast, axis=-1)

        if forecast.ndim != 3:
            raise ValueError(
                f"moirai_moe: expected forecast ndim 3 after layout fix, got shape {forecast.shape}"
            )

        pred_len = int(forecast.shape[1])
        if pred_len > forecast_horizon:
            forecast = forecast[:, :forecast_horizon, :]
        elif pred_len < forecast_horizon:
            raise ValueError(
                "moirai_moe: forecast length "
                f"{pred_len} shorter than required horizon {forecast_horizon}"
            )

        return forecast
