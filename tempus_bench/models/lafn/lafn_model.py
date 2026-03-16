import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import os
import sys
from typing import (
    Optional,
    NamedTuple,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
    Any,
    Sequence,
)

from tempus_bench.models.base_model import BaseModel, PydanticBaseModel

from chronarium import Chronarium
from tempus_bench.models.lafn.lafn_inferer import DataProcessor
from tempus_bench.models.lafn.lafn_inferer import LAFNInferer


class LafnHyperparams(PydanticBaseModel):
    pass


class LafnModel(BaseModel):
    """Chronarium-backed Large Adaptive Forecasting Network (Hybrid)."""

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):

        super().__init__(params, settings, LafnHyperparams)

        manager = Chronarium(
            bucket_name=self.bucket_name,
            project=self.project,
            credentials_path=self.credentials_path,
        )

        self.model = manager.load_remote_model(
            model_name=self.model_name,
            model_version=self.model_version,
        )
        remote_prefix = manager._remote_repo.remote_version_prefix(
            self.model_name, self.model_version
        )

        model_params = manager._remote_repo._download_model_config(remote_prefix)

        data_processor = DataProcessor(
            scaling_method="z_score",
            transform_method="arcsinh",
            in_length=model_params["in_length"],
            out_length=model_params["out_length"],
            in_features=model_params["num_y_features"],
            out_features=model_params["num_y_features"],
        )

        self.inferer = LAFNInferer(model=self.model, data_processor=data_processor)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> "LafnModel":
        """Pre-trained model – no fine-tuning required."""

        return self

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        self.model.eval()

        forecast_horizon = timestamps_target.shape[0]
        num_forecast_features = y_context.shape[-1]

        y_context = np.expand_dims(y_context, axis=0)
        timestamps_context = np.expand_dims(timestamps_context, axis=(0, -1))
        timestamps_target = np.expand_dims(timestamps_target, axis=(0, -1))

        forecasts = self.inferer.forecast(
            context_y=y_context,
            forecast_horizon=forecast_horizon,
        )

        forecasts = np.asarray(forecasts)

        samples = self.inferer.sample(
            context_y=y_context,
            forecast_horizon=forecast_horizon,
            num_samples=kwargs["num_samples"],
        )

        samples = np.asarray(samples)

        return forecasts, samples
