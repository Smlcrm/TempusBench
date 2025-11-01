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

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> "LafnModel":
        """Pre-trained model – no fine-tuning required."""

        return self

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        self.model.eval()

        forecast_horizon = timestamps_target.shape[0]
        num_forecast_features = y_context.shape[-1]

        y_context = np.expand_dims(y_context, axis=0)
        timestamps_context = np.expand_dims(timestamps_context, axis=(0, -1))
        timestamps_target = np.expand_dims(timestamps_target, axis=(0, -1))

        forecasts = self.model.forecast(
            context_y=y_context,
            context_x=timestamps_context,
            context_target=timestamps_target,
        )
        forecasts = forecasts[:, :forecast_horizon, :num_forecast_features]
        forecasts = jnp.squeeze(forecasts, axis=0)
        forecasts = np.asarray(forecasts)

        samples = self.model.sample(
            context_y=y_context,
            context_x=timestamps_context,
            context_target=timestamps_target,
            num_samples=self.num_samples,
        )
        samples = samples[:, :forecast_horizon, :num_forecast_features]
        samples = jnp.squeeze(samples, axis=1)
        samples = np.asarray(samples)

        return forecasts, samples
