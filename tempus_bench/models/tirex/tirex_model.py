from typing import Any, Dict

import numpy as np
import torch
from pydantic import BaseModel as PydanticBaseModel

from tempus_bench.models.base_model import BaseModel, validate_inputs


class TirexHyperparams(PydanticBaseModel):
    pass


class TirexModel(BaseModel):
    """
    TiRex zero-shot forecasting model with enhanced in-context learning.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TiRex model wrapper.

        Args:
            params: Model parameters dictionary
            settings: Model configuration settings
        """
        super().__init__(params, settings, TirexHyperparams)
        
        self._model = None
        self.backend = getattr(self, 'backend', 'torch')
        
    def _load_model(self):
        """Load TiRex model from HuggingFace."""
        if self._model is None:
            from tirex import load_model
            
            hf_model_name = self.hf_model_name
            backend = self.backend
            
            self._model = load_model(hf_model_name, backend=backend)
            
            if backend != "cuda" and hasattr(self._model, 'to'):
                device = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
                self._model.to(device)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "TirexModel":
        """
        Initialize TiRex model for zero-shot forecasting.

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
        Generate predictions using TiRex model.

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
            Input shape (num_targets, num_steps) transposed to match TempusBench format.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained (initialized) before prediction")
            
        self._load_model()
        
        forecast_horizon = timestamps_target.shape[0]
        
        if y_context.ndim == 1:
            y_context = y_context.reshape(1, -1)
            is_univariate = True
        elif y_context.ndim == 2:
            y_context = y_context.T
            is_univariate = False
        else:
            raise ValueError(f"y_context must be 1D or 2D, got shape {y_context.shape}")
        
        context_tensor = torch.tensor(y_context, dtype=torch.float32)
        
        quantiles, mean = self._model.forecast(
            context=context_tensor,
            prediction_length=forecast_horizon
        )
        
        quantiles_np = quantiles.cpu().numpy() if isinstance(quantiles, torch.Tensor) else quantiles
        mean_np = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
        
        # Transpose to (num_quantiles, forecast_horizon, num_targets)
        samples = np.transpose(quantiles_np, (1, 2, 0))
        
        if is_univariate:
            if samples.shape[-1] != 1:
                samples = samples[..., np.newaxis]
        
        return samples

