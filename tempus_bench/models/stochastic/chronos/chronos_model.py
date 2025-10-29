"""
Chronos foundation model implementation for time series forecasting.

This module provides a wrapper around the Amazon Chronos foundation model for time series
forecasting. Chronos is a large language model specifically designed for time series
forecasting tasks and can handle both univariate and multivariate data.

The model supports multiple sizes (tiny, mini, small, base, large) and can be configured
with different context lengths and sampling strategies.
"""

import pdb
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional
from chronos import ChronosPipeline as BaseChronosPipeline
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Literal
from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronosHyperparams(PydanticBaseModel):
    model_size: Optional[Literal["tiny", "mini", "small", "base", "large"]] = Field(default="tiny", description="Size of the Chronos model")
    context_length: Optional[int] = Field(default=2048, ge=1, description="Number of past time steps for context")

class ChronosModel(BaseModel):
    """
    Chronos foundation model wrapper for time series forecasting.

    This class provides a unified interface for the Amazon Chronos model, which is
    a large language model specifically designed for time series forecasting.

    Attributes:
        model_size: Size of the Chronos model ('tiny', 'mini', 'small', 'base', 'large')
        context_length: Number of past time steps used as context
        num_samples: Number of predictive samples to generate
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize the Chronos model wrapper.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, ChronosHyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "ChronosModel":
        """
        Initialize the Chronos model (no training required for foundation models).

        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            timestamps_context: Timestamps for y_context (not used)
            timestamps_target: Timestamps for y_target (not used)
            **kwargs: Additional keyword arguments

        Returns:
            self: The model instance

        Note:
            Chronos is a pre-trained foundation model that doesn't require training.
            This method just marks the model as ready for inference.
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        model_size = self.model_size
        context_length = self.context_length
        
        # For foundation models, we don't need to load the model here
        # It will be loaded fresh for each prediction (like it was in the working version)
        # Load the Chronos model fresh for each prediction (like the working version)
        hf_model_name = f"amazon/chronos-t5-{model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("ChronosModel.train", f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        self._model = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.logger.info("ChronosModel.train", "Chronos model loaded successfully!")

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
        Make predictions using the trained Chronos model.

        Args:
            y_context: Recent target values for context
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model predictions

        Raises:
            ValueError: If model is not fitted or required data is missing
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        num_samples = kwargs["num_samples"]
        
        # Reference params, settings, device, python_version
        model_size = self.model_size
        context_length = self.context_length

        forecast_horizon = timestamps_target.shape[0]

        padding_length = context_length - y_context.shape[0]
        if padding_length <= 0:
            # Use the most recent context_length data points
            y_context = y_context[-context_length :, :]
        else:
            # If not enough data, pad with the last available value
            y_context = np.pad(
                y_context,
                ((padding_length, 0), (0, 0)),
                mode="constant"
            )

        y_context = torch.tensor(y_context.T)
        # Generate forecasts
        forecasts = self._model.predict(
            context=y_context,
            prediction_length=forecast_horizon,
            num_samples=self.eval_config["num_samples"]
        )
        self.logger.debug("ChronosModel.predict", f"Shape of Stochastic Forecasts {forecasts.shape}")
        forecasts = np.asarray(forecasts)

        # Chronos returns shape (num_targets, num_samples, forecast_horizon)
        # We need to transpose to (num_samples, forecast_horizon, num_targets)
        forecasts = np.transpose(forecasts, (1, 2, 0))
        return forecasts

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Chronos model's properties.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            "model_type": "Chronos",
            "model_size": self._model_config["model_size"],
            "context_length": self._model_config["context_length"],
            "num_samples": self.num_samples,
            "forecast_horizon": self.forecast_horizon,
            "is_fitted": self.is_fitted,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
