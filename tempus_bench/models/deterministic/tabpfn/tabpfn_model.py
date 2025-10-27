import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import os
import math
from tabpfn import TabPFNRegressor
from tempus_bench.models.base_model import BaseModel
import torch


def make_time_features(n: int) -> pd.DataFrame:
    """
    Produce basic cyclic time features for positions 0..n-1.
    Mirrors TabPFN-TS style feature engineering for univariate forecasting.
    """
    t = np.arange(n)
    features = {
        "t": t,
        "sin_1": np.sin(2 * np.pi * t / max(1, n)),
        "cos_1": np.cos(2 * np.pi * t / max(1, n)),
        "sin_2": np.sin(4 * np.pi * t / max(1, n)),
        "cos_2": np.cos(4 * np.pi * t / max(1, n)),
    }
    return pd.DataFrame(features)


class TabpfnModel(BaseModel):

    def __init__(self, config: UnifiedConfig, logs_path: str):
        """
        Initializes a TabPFN-TS forecaster

        Args:
            config: Configuration dictionary containing model parameters
            logs_path: Directory for storing log files (optional)
        """
        super().__init__(config_path, logs_path, hyperparameters)

        # self.model_config["allow_large_cpu_dataset"]
        # self.model_config["max_sequence_length"]

        # Set device - default to CPU for TabPFN
        # self.device = model_config.get("device", "cpu")
        self.model = None
        self.is_fitted = False

    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "TabpfnModel":
        # Zero-shot TabPFN uses context during predict; mark as fitted

        self.is_fitted = True
        return self

    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):
        # Map legacy keys to expected ones for backward compatibility
        context_window = (
            int(self.model_config.get("context_window")
                if self.model_config.get("context_window") is not None
                else self.model_config.get("max_sequence_length"))
        )
        forecast_window = (
            int(self.model_config.get("forecast_window")
                if self.model_config.get("forecast_window") is not None
                else self.model_config.get("prediction_length"))
        )

        # Determine total horizon from target timestamps
        forecast_horizon = int(getattr(timestamps_target, "shape", [0])[0])

        # Ensure 1D arrays for context/targets
        y_context = np.squeeze(y_context).astype(np.float32)

        # Use last context_window points
        y_hist = y_context[-context_window:]

        # Build time features and fit TabPFN on the context window
        X_hist = make_time_features(len(y_hist)).values
        regressor = TabPFNRegressor()
        print("Fitting TabPFN")
        regressor.fit(X_hist, y_hist)

        # Roll out forecasts in chunks
        preds: list[np.ndarray] = []
        remaining = forecast_horizon
        while remaining > 0:
            step = min(forecast_window, remaining)
            # Generate future feature positions immediately following history
            X_future = make_time_features(len(y_hist) + step).values[-step:]
            y_step = regressor.predict(X_future)
            y_step = np.asarray(y_step, dtype=np.float32)
            preds.append(y_step.reshape(-1, 1))
            # Autoregressively extend history
            y_hist = np.concatenate([y_hist, y_step])
            remaining -= step

        return np.concatenate(preds, axis=0)

    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ) -> "TabpfnModel":
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            self.models = []
            num_targets = y_context.shape[1]
            for k in range(num_targets):
                m = TabpfnModel(self.config_path, logs_path=self.logs_path, hyperparameters=self.model_config)
                yc = y_context[:, k]
                yt = y_target[:, k] if (y_target is not None and y_target.ndim > 1 and y_target.shape[1] > k) else y_target
                m._train(y_context=yc, y_target=yt, timestamps_context=timestamps_context, timestamps_target=timestamps_target, freq=freq, **kwargs)
                self.models.append(m)
            self.is_fitted = True
            return self
        return self._train(y_context, y_target, timestamps_context, timestamps_target, freq, **kwargs)

    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        freq: str,
        **kwargs,
    ):
        if hasattr(self, "models") and self.models:
            preds = []
            for k, m in enumerate(self.models):
                yc = y_context[k, :] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(y_context=yc, timestamps_context=timestamps_context,
                                timestamps_target=timestamps_target, freq=freq, **kwargs)
                preds.append(pk.reshape(-1, 1) if pk.ndim == 1 else pk)
            return np.concatenate(preds, axis=1)
        return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)
