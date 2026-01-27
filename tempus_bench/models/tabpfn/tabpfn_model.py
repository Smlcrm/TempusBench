from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field

try:
    from tabpfn import TabPFNRegressor
except ImportError as e:
    raise ImportError(
        "Failed to import TabPFNRegressor from tabpfn. "
        "Please ensure tabpfn is installed in the conda environment. "
        "Install with: pip install tabpfn==2.1.2"
    ) from e

from tempus_bench.models.base_model import BaseModel, validate_inputs


class TabpfnHyperparams(PydanticBaseModel):
    pass

class TabpfnModel(BaseModel):

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, TabpfnHyperparams)

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
    ) -> "TabpfnModel":
        # Zero-shot TabPFN uses context during predict; mark as fitted
        self.is_fitted = True
        return self

    @validate_inputs
    def _predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict
    ):
        # Map legacy keys to expected ones for backward compatibility
        context_window = int(kwargs.get("context_window", self.max_sequence_length))
        forecast_window = int(kwargs.get("forecast_window", kwargs.get("prediction_length", self.max_sequence_length)))

        # Determine total horizon from target timestamps
        forecast_horizon = int(getattr(timestamps_target, "shape", [0])[0])

        # Ensure 1D arrays for context/targets
        y_context = np.squeeze(y_context).astype(np.float32)

        # Use last context_window points
        y_hist = y_context[-context_window:]

        # Build time features and fit TabPFN on the context window
        X_hist = make_time_features(len(y_hist)).values
        regressor = TabPFNRegressor()
        regressor.fit(X_hist, y_hist)

        # Roll out forecasts in chunks
        preds: list[np.ndarray] = []
        remaining = forecast_horizon
        while remaining > 0:
            step = min(forecast_window, remaining)
            # Generate future feature positions immediately following history
            X_future = make_time_features(len(y_hist) + step).values[-step:]
            y_step = regressor.predict(X_future)
            y_step = np.asarray(y_step, dtype=np.float32).flatten()
            preds.append(y_step)
            # Autoregressively extend history
            y_hist = np.concatenate([y_hist, y_step])
            remaining -= step

        # Concatenate all prediction steps and ensure shape (forecast_horizon, 1)
        return np.concatenate(preds, axis=0).reshape(-1, 1)

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict
    ):
        forecast_horizon = timestamps_target.shape[0]
        num_targets = y_context.shape[1]
        preds = np.zeros((forecast_horizon, num_targets), dtype=np.float32)
        for k in range(num_targets):
            yc = y_context[:, k:k+1]
            pk = self._predict(y_context=yc, timestamps_context=timestamps_context,
                               timestamps_target=timestamps_target,
                               x_context=x_context, x_target=x_target, **kwargs)
            preds[:, k:k+1] = pk
        return preds

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