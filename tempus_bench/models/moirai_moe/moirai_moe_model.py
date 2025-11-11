from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel, Field
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from tempus_bench.models.base_model import BaseModel, validate_inputs


class MoiraiMoeHyperparams(PydanticBaseModel):
    pass


class MoiraiMoeModel(BaseModel):
    # Class-level cache to store loaded MoiraiMoEModule by model path
    # This prevents repeated API calls to HuggingFace when loading the same model
    _module_cache: Dict[str, MoiraiMoEModule] = {}

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
        self._prediction_length = None  # Store prediction_length to check if model needs recreation
        self._context_length = None  # Store context_length to check if model needs recreation

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
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
            pdt = y_target.shape[0]
            ctx = y_context.shape[0]
            model_path = f"Salesforce/moirai-moe-1.0-R-{self.model_size}"

            # Check if module is already cached (class-level cache to avoid repeated HF API calls)
            # This prevents rate limiting when processing many windows/tasks
            if model_path not in MoiraiMoeModel._module_cache:
                MoiraiMoeModel._module_cache[model_path] = MoiraiMoEModule.from_pretrained(
                    pretrained_model_name_or_path=model_path
                )

            # Reuse cached module instance
            cached_module = MoiraiMoeModel._module_cache[model_path]
            # Calculate patch_size: use context + prediction length
            # MoiraiMoEForecast doesn't support "auto", so we calculate it
            patch_size = ctx + pdt
            self._model = MoiraiMoEForecast(
                module=cached_module,
                prediction_length=pdt,
                context_length=ctx,
                patch_size=patch_size,
                num_samples=kwargs["num_samples"],
                target_dim=y_context.shape[1],
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self._prediction_length = pdt  # Store for later comparison
            self._context_length = ctx  # Store for later comparison
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
        forecast_horizon = timestamps_target.shape[0]
        ctx = y_context.shape[0]
        num_samples = kwargs.get("num_samples", 100)
        
        # Debug: Log the forecast horizon to verify it's correct
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"MoiraiMoE predict: forecast_horizon={forecast_horizon}, ctx={ctx}, num_targets={num_targets}")
        
        # Always create a fresh MoiraiMoEForecast instance in predict() with the correct forecast horizon
        # MoiraiMoEForecast requires prediction_length to match the actual forecast horizon
        # The forecast horizon can vary between different windows/tasks, so we must create a new
        # model instance each time to ensure prediction_length matches the actual forecast horizon
        # Note: We cache the underlying MoiraiMoEModule (which is expensive to load), but we create
        # a fresh MoiraiMoEForecast wrapper each time since it needs different prediction_length
        model_path = f"Salesforce/moirai-moe-1.0-R-{self.model_size}"
        
        # Get cached module (should already be cached from train())
        if model_path not in MoiraiMoeModel._module_cache:
            MoiraiMoeModel._module_cache[model_path] = MoiraiMoEModule.from_pretrained(
                pretrained_model_name_or_path=model_path
            )
        
        cached_module = MoiraiMoeModel._module_cache[model_path]
        # Calculate patch_size: use context + prediction length to ensure model can output full prediction
        # MoiraiMoEForecast doesn't support "auto", so we calculate it
        # Use the full context + prediction length to ensure the model can handle the full forecast horizon
        patch_size = ctx + forecast_horizon
        
        # Create a fresh MoiraiMoEForecast instance with the correct prediction_length
        # Do NOT reuse self._model from training - it has the wrong prediction_length
        fresh_model = MoiraiMoEForecast(
            module=cached_module,
            prediction_length=forecast_horizon,  # Use actual forecast horizon
            context_length=ctx,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=num_targets,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        
        # MoiraiMoEForecast requires prediction_length to be set on hparams after creation
        # This is how the model actually determines the output length
        if hasattr(fresh_model, 'hparams') and hasattr(fresh_model.hparams, 'prediction_length'):
            fresh_model.hparams.prediction_length = forecast_horizon
        
        # Debug: Verify the model's prediction_length is actually set correctly
        actual_model_prediction_length = None
        if hasattr(fresh_model, 'hparams') and hasattr(fresh_model.hparams, 'prediction_length'):
            actual_model_prediction_length = fresh_model.hparams.prediction_length
        elif hasattr(fresh_model, 'prediction_length'):
            actual_model_prediction_length = fresh_model.prediction_length
        
        if actual_model_prediction_length is not None and actual_model_prediction_length != forecast_horizon:
            # Force update if it's still wrong
            if hasattr(fresh_model, 'hparams'):
                fresh_model.hparams.prediction_length = forecast_horizon
            if hasattr(fresh_model, 'prediction_length'):
                fresh_model.prediction_length = forecast_horizon
        
        # Use the fresh model for prediction (don't store it in self._model to avoid confusion)
        self._prediction_length = forecast_horizon  # Store for later comparison
        self._context_length = ctx  # Store for later comparison
        
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

        # Use the fresh model instance created above (not self._model from training)
        forecast = fresh_model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        # MoiraiMoEForecast returns shape: (batch, num_samples, prediction_length, num_targets)
        # Convert to numpy array
        forecast = np.asarray(forecast)
        
        # Handle 4D output: (batch, num_samples, prediction_length, num_targets)
        # Remove batch dimension if present
        if forecast.ndim == 4:
            # Shape: (batch, num_samples, prediction_length, num_targets)
            # Squeeze batch dimension: (num_samples, prediction_length, num_targets)
            forecast = np.squeeze(forecast, axis=0)
        elif forecast.ndim == 3:
            # Shape might be: (num_targets, num_samples, prediction_length) or (num_samples, prediction_length, num_targets)
            # Check if first dimension is num_targets (should be smaller than num_samples typically)
            if forecast.shape[0] == num_targets and forecast.shape[0] < forecast.shape[1]:
                # Shape is (num_targets, num_samples, prediction_length)
                # Transpose to (num_samples, prediction_length, num_targets)
                forecast = np.transpose(forecast, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected forecast output shape: {forecast.ndim}D with shape {forecast.shape}")
        
        # Now forecast should be (num_samples, prediction_length, num_targets)
        # Handle prediction_length mismatch: MoiraiMoEForecast may output a different length
        # than requested, so we pad or truncate to match the expected forecast_horizon
        actual_prediction_length = forecast.shape[1]
        if actual_prediction_length != forecast_horizon:
            # Pad or truncate to match expected forecast_horizon
            if actual_prediction_length < forecast_horizon:
                # Pad with the last value
                padding_shape = (forecast.shape[0], forecast_horizon - actual_prediction_length, forecast.shape[2])
                padding = np.broadcast_to(
                    forecast[:, -1:, :],  # Last timestep
                    padding_shape
                )
                forecast = np.concatenate([forecast, padding], axis=1)
            elif actual_prediction_length > forecast_horizon:
                # Truncate to expected length
                forecast = forecast[:, :forecast_horizon, :]
        
        # Final shape should be: (num_samples, prediction_length, num_targets)
        # Example: (100, 16, 10)

        return forecast
