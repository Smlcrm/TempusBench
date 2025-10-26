"""
Chronos foundation model implementation for time series forecasting.

This module provides a wrapper around the Amazon Chronos foundation model for time series
forecasting. Chronos is a large language model specifically designed for time series
forecasting tasks and can handle both univariate and multivariate data.

The model supports multiple sizes (tiny, mini, small, base, large) and can be configured
with different context lengths and sampling strategies.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, List, Optional
from tempus_bench.models.stochastic_base_model import StochasticBaseModel
from tempus_bench.utils.logger import get_logger
from chronos import ChronosPipeline as BaseChronosPipeline
from einops import rearrange


class ChronosModel(StochasticBaseModel):
    """
    Chronos foundation model wrapper for time series forecasting.

    This class provides a unified interface for the Amazon Chronos model, which is
    a large language model specifically designed for time series forecasting.

    Attributes:
        model_size: Size of the Chronos model ('tiny', 'mini', 'small', 'base', 'large')
        context_length: Number of past time steps used as context
        num_samples: Number of predictive samples to generate
        pipeline: The underlying Chronos pipeline instance
    """

    def __init__(self, config: Dict[str, Any], logs_dir: str):
        """
        Initialize the Chronos model wrapper.

        Args:
            config: Configuration dictionary containing model parameters
                - model_size: str, size of the Chronos model (default: 'small')
                - context_length: int, number of past time steps for context (default: 8)
                - num_samples: int, number of predictive samples (default: 5)
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, logs_dir)

        self.model_config["model_size"] = (
            "tiny"  # Valid model sizes = {'tiny', 'mini', 'small', 'base', 'large'}
        )
        self.model_config["context_length"] = 512

        # Initialize model state
        self.is_fitted = False

        # forecast_horizon is inherited from parent class (FoundationModel)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
    ) -> "ChronosModel":
        """
        Initialize the Chronos model (no training required for foundation models).

        Args:
            y_context: Past target values (not used for training, for compatibility)
            y_target: Future target values (not used for training, for compatibility)
            y_start_date: Start date for y_context (not used)

        Returns:
            self: The model instance

        Note:
            Chronos is a pre-trained foundation model that doesn't require training.
            This method just marks the model as ready for inference.
        """
        # For foundation models, we don't need to load the model here
        # It will be loaded fresh for each prediction (like it was in the working version)
        # Load the Chronos model fresh for each prediction (like the working version)
        hf_model_name = f"amazon/chronos-t5-{self.model_config['model_size']}"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("ChronosModel", f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        self.model = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.logger.info("ChronosModel", "Chronos model loaded successfully!")

        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray = None,
        timestamps_context: np.ndarray = None,
        timestamps_target: np.ndarray = None,
        freq: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the trained Chronos model.

        Args:
            y_context: Recent target values for context
            y_target: Target values to predict (used to determine forecast length)
            y_context_timestamps: Timestamps for context data
            y_target_timestamps: Timestamps for target data
            forecast_horizon: Number of steps to forecast (overrides y_target length if provided)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Model prediction samples with shape (num_samples, forecast_horizon, num_targets)

        Raises:
            ValueError: If model is not fitted or required data is missing
        """

        forecast_horizon = timestamps_target.shape[0]

        padding_length = self.model_config["context_length"] - y_context.shape[0]
        if padding_length <= 0:
            # Use the most recent context_length data points
            y_context = y_context[-self.model_config["context_length"] :, :]
        else:
            # If not enough data, pad with the last available value
            y_context = np.pad(
                y_context,
                ((padding_length, 0), (0, 0)),
                mode="constant"
            )

        y_context = torch.tensor(y_context.T)
        # Generate forecasts
        forecasts = self.model.predict(
            context=y_context,
            prediction_length=forecast_horizon,
            num_samples=self.num_samples,
        )
        forecasts = np.squeeze(np.asarray(forecasts))
        
        # Ensure correct shape: (num_samples, forecast_horizon, num_targets)
        if forecasts.ndim == 2:
            # If univariate, add target dimension
            forecasts = forecasts[:, :, np.newaxis]
        
        return forecasts

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Chronos model's properties.

        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            "model_type": "Chronos",
            "model_size": self.model_config["model_size"],
            "context_length": self.model_config["context_length"],
            "num_samples": self.num_samples,
            "forecast_horizon": self.forecast_horizon,
            "is_fitted": self.is_fitted,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
