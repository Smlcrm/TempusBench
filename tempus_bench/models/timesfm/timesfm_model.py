import os
import numpy as np
import pandas as pd
import timesfm
import torch

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs


class TimesfmHyperparams(PydanticBaseModel):
    pass


class TimesfmModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TimesFM model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, TimesfmHyperparams)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "TimesfmModel":
        """
        Foundation model: no training needed. Mark as fitted and return self.
        """
        self._build_model()
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
        if not self.is_fitted:
            raise ValueError("TimesFMModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        predictions = self._model.forecast(y_context.T)[0].T[:forecast_horizon]
        return predictions

    def _build_model(self):
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self.device,
                input_patch_len=self.input_patch_len,
                horizon_len=self.horizon_len,
                num_layers=self.num_layers,
                model_dims=self.model_dims,
                # Se this to True for v1.0 checkpoints
                output_patch_len=self.output_patch_len,
                use_positional_embedding=self.use_positional_embedding,
                context_len=self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path=None,
                version=self.version,
                huggingface_repo_id=self.huggingface_repo_id,
                local_dir=os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "checkpoints")
                ),
            ),
        )
        self.is_fitted = True
