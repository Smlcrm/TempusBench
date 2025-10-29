import numpy as np
import pandas as pd

from typing import Dict, Any, Optional
from tabpfn import TabPFNRegressor
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs


class TabpfnHyperparams(PydanticBaseModel):
    allow_large_cpu_dataset: Optional[bool] = Field(default=False, description="Allow large CPU datasets")
    max_sequence_length: Optional[int] = Field(default=1000, ge=1, description="Maximum sequence length")
    device: Optional[str] = Field(default="cpu", description="Device to use for computation")


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

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initializes a TabPFN-TS forecaster with model-specific parameters
        """
        super().__init__(params, settings, TabpfnHyperparams)

        # Set device - default to CPU for TabPFN
        self.device = self.device
        self._model = None
        self.is_fitted = False

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
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
        self.logger.info("TabPFNModel.predict", "Fitting TabPFN")
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

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ) -> "TabpfnModel":
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        max_sequence_length = self.max_sequence_length
        
        if y_context.ndim > 1 and y_context.shape[1] > 1:
            self._models = []
            num_targets = y_context.shape[1]
            for k in range(num_targets):
                m = TabpfnModel(params=self.dict(), settings=self.settings)
                yc = y_context[:, k]
                yt = y_target[:, k] if (y_target is not None and y_target.ndim > 1 and y_target.shape[1] > k) else y_target
                m._train(y_context=yc, y_target=yt, timestamps_context=timestamps_context, timestamps_target=timestamps_target, **kwargs)
                self._models.append(m)
            self.is_fitted = True
            return self
        return self._train(y_context, y_target, timestamps_context, timestamps_target, **kwargs)

    @validate_inputs
    def predict(
        self,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        **kwargs: dict
    ):
        # Extract kwargs (NO defaults, use kwargs["var_name"])
        freq = kwargs["freq"]
        
        # Reference params, settings, device, python_version
        max_sequence_length = self.max_sequence_length
        
        if hasattr(self, "models") and self._models:
            preds = []
            for k, m in enumerate(self._models):
                yc = y_context[k, :] if y_context is not None and y_context.ndim > 1 else y_context
                pk = m._predict(y_context=yc, timestamps_context=timestamps_context,
                                timestamps_target=timestamps_target, **kwargs)
                preds.append(pk.reshape(-1, 1) if pk.ndim == 1 else pk)
            return np.concatenate(preds, axis=1)
        return self._predict(y_context, timestamps_context, timestamps_target, **kwargs)
