import os
import subprocess
import sys
import warnings

# Suppress FutureWarnings/UserWarnings from gluonts and pytorch (third-party, not fixable here)
# Must be before any gluonts/lightning imports
warnings.filterwarnings(
    "ignore",
    message=".*is deprecated and will be removed in a future version.*",
    category=FutureWarning,
    module="gluonts.time_feature",
)
warnings.filterwarnings(
    "ignore",
    message=".*non-tuple sequence for multidimensional indexing.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Using `json`-module for json-handling.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=DeprecationWarning,
)

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from pydantic import BaseModel as PydanticBaseModel, Field
import torch
from tempus_bench.compat.lightning_pytree import apply_lightning_pytree_leafspec_patch
from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support
from tempus_bench.utils.lagllama_freq import normalize_freq_for_lagllama

# Add the lagllama directory to the Python path for absolute imports
lagllama_dir = os.path.dirname(os.path.abspath(__file__))
if lagllama_dir not in sys.path:
    sys.path.insert(0, lagllama_dir)

apply_lightning_pytree_leafspec_patch()

from lag_llama.gluon.estimator import LagLlamaEstimator


def _lagllama_checkpoint_file(snapshot_dir: str) -> str:
    """Pick a Lightning ``.ckpt`` under an HF snapshot / FUSE directory."""
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"Lag-Llama weights directory does not exist: {snapshot_dir!r}")
    ckpts = [os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt files under Lag-Llama weights directory {snapshot_dir!r}"
        )
    preferred = [p for p in ckpts if "pretrained" in os.path.basename(p).lower()]
    if preferred:
        return preferred[0]
    return sorted(ckpts)[0]


def _normalize_freq(freq: str) -> str:
    """Delegate to :func:`normalize_freq_for_lagllama` (defined in ``freq_utils`` for testability)."""
    return normalize_freq_for_lagllama(freq)


class LagllamaHyperparams(PydanticBaseModel):
    pass


# Try to import lag_llama, install if not available
class LagllamaModel(BaseModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Lag-Llama model with BaseModel interface.

        Args:
            params: Model parameters dictionary
            settings: Settings dictionary containing device, python_version, etc.
        """
        # Initialize base model
        super().__init__(params, settings, LagllamaHyperparams)

    def _create_predictor_for_horizon(
        self, forecast_horizon: int, num_samples: int = 10
    ):
        """Create a predictor for a specific forecast horizon.

        Always uses input_size=1 (univariate). Multivariate targets and covariates
        are handled by iterating: one univariate call per variate.

        Args:
            forecast_horizon: Number of future time steps to predict
            num_samples: Number of probabilistic samples

        Returns:
            Predictor instance
        """
        context_length = self.context_length
        batch_size = self.batch_size

        ckpt_path = None
        hf_or_dir = getattr(self, "hf_model_name", None)
        if hf_or_dir and os.path.isdir(hf_or_dir):
            ckpt_path = _lagllama_checkpoint_file(hf_or_dir)

        estimator = LagLlamaEstimator(
            prediction_length=forecast_horizon,
            context_length=context_length,
            input_size=1,
            batch_size=batch_size,
            num_parallel_samples=num_samples,
            device=torch.device(self.device),
            ckpt_path=ckpt_path,
        )

        transformation = estimator.create_transformation()
        lightning_module = estimator.create_lightning_module()
        predictor = estimator.create_predictor(transformation, lightning_module)

        return predictor

    def convert_to_datetimeindex(self, timestamps):
        # Convert timestamps to datetime if they're not already
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(timestamps[0], (int, np.integer)):
                min_ts = np.min(timestamps)
                max_ts = np.max(timestamps)

                # Pandas datetime bounds for 64-bit ns: 1677-09-21 to 2262-04-11
                # 1677-09-21T00:12:43.145224192Z = -9223372036854775808 ns
                # 2262-04-11T23:47:16.854775807Z = 9223372036854775807 ns
                NS_LOWER = -9223372036854775808
                NS_UPPER = 9223372036854775807
                US_LOWER = NS_LOWER // 1000
                US_UPPER = NS_UPPER // 1000
                MS_LOWER = NS_LOWER // 1_000_000
                MS_UPPER = NS_UPPER // 1_000_000
                S_LOWER = NS_LOWER // 1_000_000_000
                S_UPPER = NS_UPPER // 1_000_000_000

                def in_bounds(val, lower, upper):
                    return lower <= val <= upper

                # Try to classify the likely unit and check bounds
                unit = None
                if isinstance(min_ts, (int, np.integer)):
                    # Try nanoseconds
                    if in_bounds(min_ts, NS_LOWER, NS_UPPER) and in_bounds(
                        max_ts, NS_LOWER, NS_UPPER
                    ):
                        unit = "ns"
                    # Try microseconds
                    elif in_bounds(min_ts, US_LOWER, US_UPPER) and in_bounds(
                        max_ts, US_LOWER, US_UPPER
                    ):
                        unit = "us"
                    # Try milliseconds
                    elif in_bounds(min_ts, MS_LOWER, MS_UPPER) and in_bounds(
                        max_ts, MS_LOWER, MS_UPPER
                    ):
                        unit = "ms"
                    # Try seconds
                    elif in_bounds(min_ts, S_LOWER, S_UPPER) and in_bounds(
                        max_ts, S_LOWER, S_UPPER
                    ):
                        unit = "s"
                    else:
                        raise ValueError(
                            f"Timestamps are out of bounds for pandas datetime64[ns] (min={min_ts}, max={max_ts})."
                        )
                    timestamps = pd.to_datetime(timestamps, unit=unit)
                else:
                    timestamps = pd.to_datetime(timestamps)

        return timestamps


    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions for a single univariate variate.

        Args:
            y_context: Single variate (num_steps, 1)
            timestamps_context: Timestamps for context data
            timestamps_target: Timestamps for target data

        Returns:
            np.ndarray: Model prediction samples (num_samples, forecast_horizon, 1)
        """
        freq = _normalize_freq(str(kwargs["freq"]))
        num_samples = kwargs["num_samples"]

        if y_context.ndim == 1:
            y_context = y_context[:, np.newaxis]
        elif y_context.shape[1] != 1:
            raise ValueError(
                f"_predict only handles univariate data. Got y_context with shape {y_context.shape}."
            )

        start_time = self.convert_to_datetimeindex(timestamps_context)[0]
        periods = y_context.shape[0]
        timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)

        context_df = pd.DataFrame(
            {"ds": timestamps, "target": y_context[:, 0].astype("float32"), "unique_id": "test_series"},
            index=range(periods),
        )
        dataset = PandasDataset.from_long_dataframe(
            context_df,
            target="target",
            timestamp="ds",
            item_id="unique_id",
            freq=freq,
        )

        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.predictor,
            num_samples=num_samples,
        )

        forecast = next(forecast_it)
        samples = np.asarray(forecast.samples)
        return np.expand_dims(samples, axis=2)  # (num_samples, forecast_horizon, 1)

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
    ) -> "LagllamaModel":
        """
        Lag-Llama is pre-trained. Creates a univariate predictor (input_size=1).
        Multivariate targets and covariates handled by iterating: one univariate call per variate.
        """
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Lag-Llama",
        )
        if not hasattr(self, "predictor"):
            self.predictor = self._create_predictor_for_horizon(
                y_target.shape[0],
                num_samples=kwargs["num_samples"],
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
        Make predictions using the trained Lag-Llama model.

        Iterates over M targets + N covariates (when present): one univariate call per variate.
        Returns only the M target predictions (covariate predictions are discarded).

        Returns:
            np.ndarray: Model predictions with shape (num_samples, forecast_horizon, num_targets)
        """
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="Lag-Llama",
        )
        freq = _normalize_freq(str(kwargs["freq"]))
        num_samples = kwargs["num_samples"]

        # Build variates: M targets + N covariates (one univariate call per variate)
        if x_context is not None:
            variates = np.concatenate([y_context, x_context], axis=1)
        else:
            variates = y_context if y_context.ndim == 2 else y_context[:, np.newaxis]

        num_targets = y_context.shape[1] if y_context.ndim == 2 else 1

        preds = []
        for k in range(variates.shape[1]):
            pred = self._predict(
                y_context=variates[:, k : k + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                freq=freq,
                num_samples=num_samples,
            )
            preds.append(pred)

        # Use only first M outputs (target predictions); discard covariate predictions
        return np.concatenate(preds[:num_targets], axis=2)
