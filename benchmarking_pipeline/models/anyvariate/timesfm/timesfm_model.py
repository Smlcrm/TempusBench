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

        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=forecast_horizon,
                num_layers=50,
                # Se this to True for v1.0 checkpoints
                use_positional_embedding=False,
                # Note that we could set this to as high as 2048 but keeping it 512 here so that
                # both v1.0 and 2.0 checkpoints work
                context_len=512,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-jax"
            ),
        )

        # Generate forecasts
        point_forecasts = model.forecast(
            y_context[:, 0].tolist(), timestamps_context[:, 0].tolist()
        )

        return point_forecasts
