"""
Chronos-2

References:
    Paper: https://arxiv.org/abs/2510.15821
    GitHub: https://github.com/amazon-science/chronos-forecasting
    HuggingFace: https://huggingface.co/amazon/chronos-2
"""

from typing import Any, Dict

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs


class Chronos2Hyperparams(PydanticBaseModel):
    """Hyperparameters for Chronos-2 model."""
    pass


class Chronos2Model(BaseModel):
    """
    Chronos-2 foundation model for universal forecasting.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Chronos-2 model wrapper.

        Args:
            params: Model parameters dictionary
            settings: Model configuration settings
        """
        super().__init__(params, settings, Chronos2Hyperparams)
        self._model = None

    def _load_model(self):
        """Load Chronos-2 model from HuggingFace."""
        if self._model is None:
            from chronos import Chronos2Pipeline
            
            hf_model_name = self.hf_model_name
            
            self._model = Chronos2Pipeline.from_pretrained(
                hf_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "Chronos2Model":
        """
        Initialize Chronos-2 model for zero-shot forecasting.

        Args:
            y_context: Past target values
            y_target: Future target values
            timestamps_context: Timestamps for context
            timestamps_target: Timestamps for target
            **kwargs: Additional keyword arguments

        Returns:
            Model instance

        Note:
            Pre-trained model requires no training. Loads model and marks as ready.
        """
        self._load_model()
        
        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate predictions using Chronos-2 model.

        Args:
            y_context: Context values. Shape (num_steps, num_targets) or (num_steps,)
            timestamps_context: Context timestamps
            timestamps_target: Target timestamps for forecast horizon
            **kwargs: Additional arguments (num_samples)

        Returns:
            Predictions with shape (num_samples, forecast_horizon, num_targets)

        Raises:
            ValueError: If model not fitted

        Note:
            Context padded or trimmed to match configured context_length.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained (initialized) before prediction")
            
        self._load_model()
        
        context_length = self.context_length
        forecast_horizon = timestamps_target.shape[0]

        padding_length = context_length - y_context.shape[0]
        if padding_length <= 0:
            y_context = y_context[-context_length:, :]
        else:
            y_context = np.pad(
                y_context, ((padding_length, 0), (0, 0)), mode="constant"
            )

        y_context = torch.tensor(y_context.T)
        
        forecasts = self._model.predict(
            context=y_context, prediction_length=forecast_horizon
        )
        forecasts = np.asarray(forecasts)

        # Transpose to (num_samples, forecast_horizon, num_targets)
        forecasts = np.transpose(forecasts, (1, 2, 0))
        return forecasts

