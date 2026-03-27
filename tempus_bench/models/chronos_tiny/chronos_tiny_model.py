"""
Chronos foundation model implementation for time series forecasting.

This module provides a wrapper around the Amazon Chronos foundation model for time series
forecasting. Chronos is a large language model specifically designed for time series
forecasting tasks and can handle both univariate and multivariate data.

The model supports multiple sizes (tiny, mini, small, base, large) and can be configured
with different context lengths and sampling strategies.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline as BaseChronosPipeline
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support


class ChronosTinyHyperparams(PydanticBaseModel):
    pass


class ChronosTinyModel(BaseModel):
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
        super().__init__(params, settings, ChronosTinyHyperparams)

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
    ) -> "ChronosTinyModel":
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
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Chronos",
        )
        hf_model_name = self.hf_model_name

        # Align with settings.yaml `device` and Amazon's HF examples (cuda + bfloat16 for GPU).
        requested = str(getattr(self, "device", "cpu") or "cpu").strip().lower()
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Model settings request device=cuda but torch.cuda.is_available() is false. "
                    "Chronos-T5 Large (710M) inference is documented on GPU "
                    "(https://huggingface.co/amazon/chronos-t5-large); use a CUDA-enabled worker "
                    "image and a Google Batch GPU machine type, or set device: cpu."
                )
            device_map = "cuda"
        elif requested == "cpu":
            device_map = "cpu"
        else:
            raise ValueError(
                f"Unsupported Chronos device {requested!r}; expected 'cpu' or 'cuda'."
            )

        self._model = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
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
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Chronos",
        )
        context_length = self.context_length
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        # Extend input with x_context (past covariates only) for non-native support
        if x_context is not None:
            y_input = np.concatenate([y_context, x_context], axis=1)
        else:
            y_input = y_context

        padding_length = context_length - y_input.shape[0]
        if padding_length <= 0:
            y_input = y_input[-context_length:, :]
        else:
            y_input = np.pad(
                y_input, ((padding_length, 0), (0, 0)), mode="constant"
            )

        context_tensor = torch.tensor(y_input.T)
        forecasts = self._model.predict(
            context_tensor, prediction_length=forecast_horizon
        )
        forecasts = np.asarray(forecasts)

        # Chronos returns (num_channels, num_samples, forecast_horizon)
        # Keep only the first num_targets channels (discard covariate channels)
        forecasts = forecasts[:num_targets, :, :]
        forecasts = np.transpose(forecasts, (1, 2, 0))
        return forecasts
