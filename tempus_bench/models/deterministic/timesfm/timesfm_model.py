import os
import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel as PydanticBaseModel, Field
from tempus_bench.config.models import JobConfig
from tempus_bench.models.base_model import BaseModel, validate_inputs


class TimesfmHyperparams(PydanticBaseModel):
    # Foundation model with minimal parameters
    pass


class TimesFMModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TimesFM model.
        
        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, TimesfmHyperparams)
        
        self.is_fitted = True

        horizon = 1  # Will be set during training
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                input_patch_len=32,
                horizon_len=horizon,
                num_layers=20,
                model_dims=1280,
                # Se this to True for v1.0 checkpoints
                output_patch_len=128,
                use_positional_embedding=True,
                # Note that we could set this to as high as 2048 but keeping it 512 here so that
                # both v1.0 and 2.0 checkpoints work
                context_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path=None,
                version="jax",
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
                local_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints")),
            ),
        )

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "TimesFMModel":
        """
        Foundation model: no training needed. Mark as fitted and return self.
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        # TimesFM is a foundation model with minimal params
        
        self.is_fitted = True
        return self

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ):
        """
        Make predictions using the trained TimesFM model.
        """
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        forecast_horizon = kwargs["forecast_horizon"]
        
        # Reference params, settings, device, python_version
        # TimesFM is a foundation model with minimal params

        forecast_horizon = timestamps_target.shape[0]

        # Generate forecasts
        forecasts = self._model.forecast(y_context)[0]
        # print(forecasts)
        if len(forecasts) == 1:
            forecasts = np.expand_dims(forecasts, axis=0)

        forecasts = forecasts[:forecast_horizon]

        return forecasts
