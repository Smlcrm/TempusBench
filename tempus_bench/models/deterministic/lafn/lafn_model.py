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

from benchmarking_pipeline.models.base_model import BaseModel
from tempus_bench.config.models import UnifiedConfig
from tempus_bench.models.base_model import BaseModel

# lafn_dir = os.path.dirname(os.path.abspath(__file__))
# if lafn_dir not in sys.path:
#     sys.path.insert(0, lafn_dir)

from chronarium import Chronarium


class LAFNModel(BaseModel):
    """Chronarium-backed Large Adaptive Forecasting Network."""

    def __init__(self, config: UnifiedConfig, logs_path: str):

        super().__init__(config_path, logs_path)

        manager = Chronarium(
            bucket_name=self.bucket_name,
            project=self.project,
            credentials_path=self.credentials_path,
        )

        self.model = manager.load_remote_model(
            model_name=self.model_name,
            model_version=self.model_version,
        )

        self.data_processor = DataProcessor(
            scaling_method="none",
            in_length=2048,
            out_length=128,
            in_features=10,
            out_features=10,
        )

        self.is_fitted = True

    def train(
        self,
        y_context: Optional[np.ndarray],
        y_target: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> "LAFNModel":
        """Pre-trained model – no fine-tuning required."""
        self.is_fitted = True
        return self

    def predict(
        self,
        y_context: Optional[np.ndarray] = None,
        timestamps_context: Optional[np.ndarray] = None,
        timestamps_target: Optional[np.ndarray] = None,
        freq: str = None,
    ) -> np.ndarray:
        self.model.eval()

        forecasts = self.model.forecast(forecast_embed)
        forecasts = forecasts[:, :forecast_horizon, :num_forecast_features]
        forecasts = jnp.squeeze(forecasts, axis=0)

        return np.asarray(forecasts)
