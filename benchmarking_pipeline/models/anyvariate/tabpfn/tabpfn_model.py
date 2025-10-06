import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import os
import math
from tabpfn import TabPFNRegressor
from benchmarking_pipeline.models.base_model import BaseModel
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

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes a TabPFN-TS forecaster

        Args:
            n_ensemble_configs (int): Number of ensemble configurations (kept in signature).
            device (str): 'cpu' or 'cuda' for the underlying TabPFN model.
            allow_large_cpu_dataset (bool): If True, bypasses the default CPU sample limit by
                setting ignore_pretraining_limits=True. Otherwise will error if >1000 samples.
        """
        super().__init__(config)

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
        forecast_horizon = timestamps_target.shape[0]
        context_window = self.model_config["context_window"]
        forecast_window = self.model_config["forecast_window"]

        # Fit the model on the current context window
        regressor = TabPFNRegressor()

        timestamps_context = timestamps_context[-context_window:]
        y_context = y_context[-context_window:]

        print("Fitting TabFPN")
        regressor.fit(timestamps_context, y_context)

        y_pred = []
        steps_left = forecast_horizon

        for step in range(math.ceil(forecast_horizon / forecast_window)):

            timestamps_curr = timestamps_target[
                forecast_window * step : forecast_window * (step + 1), :
            ]

            y_pred_curr = regressor.predict(timestamps_curr)
            y_pred_curr = np.expand_dims(y_pred_curr, axis=1)
            y_pred.append(y_pred_curr)

            # Update the context and target for the next iteration
            timestamps_context = np.concatenate(
                [timestamps_context, timestamps_curr], axis=0
            )
            y_context = np.concatenate([y_context, y_pred_curr], axis=0)

        forecasts = np.concatenate(y_pred, axis=0)

        return forecasts

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
            for k in range(y_context.shape[1]):
                m = TabpfnModel(self.model_config)
                m._train(y_context[:, k], y_target[:, k] if y_target is not None and y_target.ndim > 1 else y_target,
                         timestamps_context, timestamps_target, freq, **kwargs)
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
                yc = y_context[:, k] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(y_context=yc, timestamps_context=timestamps_context,
                                timestamps_target=timestamps_target, freq=freq, **kwargs)
                preds.append(pk.reshape(-1, 1) if pk.ndim == 1 else pk)
            return np.concatenate(preds, axis=1)
        return self._predict(y_context, timestamps_context, timestamps_target, freq, **kwargs)
