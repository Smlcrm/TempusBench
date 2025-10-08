import os
import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Dict, Any, Optional, List, Union
from benchmarking_pipeline.models.base_model import BaseModel


class TimesFMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_fitted = True

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                input_patch_len=32,
                horizon_len=1000,
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

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "TimesFMModel":
        """
        Foundation model: no training needed. Mark as fitted and return self.
        """
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):

        forecast_horizon = timestamps_target.shape[0]

        # Generate forecasts
        forecasts = self.model.forecast(y_context)[0]
        # print(forecasts)
        if len(forecasts) == 1:
            forecasts = np.expand_dims(forecasts)

        forecasts = forecasts[:forecast_horizon]

        return forecasts
