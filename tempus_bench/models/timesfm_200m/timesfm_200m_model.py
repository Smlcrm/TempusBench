import os
import numpy as np
import pandas as pd
import timesfm
import torch

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs


class Timesfm200mHyperparams(PydanticBaseModel):
    pass


class Timesfm200mModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize TimesFM model.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        super().__init__(params, settings, Timesfm200mHyperparams)

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
    ) -> "Timesfm200mModel":
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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Make predictions using the trained TimesFM model.

        Uses forecast_with_covariates when covariates are provided. Supports past-only,
        future-only, or both. TimesFM requires full (context+horizon) coverage; missing
        parts are padded: past-only pads future with last observed value, future-only
        pads past with zeros (heuristic; not guaranteed optimal).
        """
        if not self.is_fitted:
            raise ValueError("TimesFMModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        if x_context is not None or x_target is not None:
            # TimesFM requires full (context + horizon) coverage; pad missing parts.
            ctx_len = y_context.shape[0]
            x_future = x_target[:forecast_horizon] if x_target is not None else None
            if x_context is not None and x_target is not None:
                full_covariates = np.concatenate([x_context, x_future], axis=0)
            elif x_context is not None:
                # Past only: pad future with last observed value
                last_row = np.tile(x_context[-1:], (forecast_horizon, 1))
                full_covariates = np.concatenate([x_context, last_row], axis=0)
            else:
                # Future only: pad past with zeros (unknown in past)
                past_pad = np.zeros((ctx_len, x_future.shape[1]), dtype=np.float64)
                full_covariates = np.concatenate([past_pad, x_future], axis=0)

            num_covariates = full_covariates.shape[1]
            inputs = [y_context[:, i].tolist() for i in range(num_targets)]
            dynamic_numerical_covariates = {
                f"cov_{cov_idx}": [full_covariates[:, cov_idx].tolist()] * num_targets
                for cov_idx in range(num_covariates)
            }

            cov_forecast, _ = self._model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                freq=[0] * num_targets,
            )
            # cov_forecast is list of (horizon,) arrays, one per target
            arr = np.array([np.asarray(x).flatten() for x in cov_forecast])
            predictions = arr.T  # (horizon, num_targets) - match non-covariate 2D format
        else:
            # Standard forecast without covariates
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
