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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
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
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Make predictions using the trained TimesFM model.

        If covariates (x_context and x_target) are provided, uses forecast_with_covariates
        to incorporate exogenous features.
        """
        if not self.is_fitted:
            raise ValueError("TimesFMModel is not fitted. Call train() first.")

        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]

        if x_context is not None and x_target is not None:
            # Combine x_context and x_target to get full covariate data
            # Shape: (context_len + horizon_len, num_covariates)
            full_covariates = np.concatenate([x_context, x_target], axis=0)
            num_covariates = full_covariates.shape[1]

            inputs = [y_context[:, i].tolist() for i in range(num_targets)]

            # Prepare dynamic numerical covariates
            # Each covariate needs to be a list of lists (one per time series)
            dynamic_numerical_covariates = {}
            for cov_idx in range(num_covariates):
                cov_values = full_covariates[:, cov_idx].tolist()
                # Replicate covariate for each time series
                dynamic_numerical_covariates[f"cov_{cov_idx}"] = [cov_values] * num_targets

            # Use forecast_with_covariates
            cov_forecast, _ = self._model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                freq=[0] * num_targets,
            )

            # Transpose back to (forecast_horizon, num_targets)
            predictions = cov_forecast.T
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
