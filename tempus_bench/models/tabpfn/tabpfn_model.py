from typing import Any, Dict, Literal, Optional

import os

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field

try:
    from tabpfn import TabPFNRegressor
except ImportError as e:
    raise ImportError(
        "Failed to import TabPFNRegressor from tabpfn. "
        "Install model deps from tempus_bench/models/tabpfn/requirements.txt "
        "(e.g. pip install 'tabpfn>=2.1.0')."
    ) from e


def _tabpfn_checkpoint_file(snapshot_dir: str) -> str:
    """Resolve a TabPFN regressor checkpoint under a FUSE/HF snapshot directory."""
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"TabPFN weights directory does not exist: {snapshot_dir!r}")
    preferred = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
    p = os.path.join(snapshot_dir, preferred)
    if os.path.isfile(p):
        return p
    ckpts = [
        os.path.join(snapshot_dir, f)
        for f in sorted(os.listdir(snapshot_dir))
        if f.endswith(".ckpt") or f.endswith(".pt")
    ]
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt/.pt TabPFN weights under {snapshot_dir!r}; "
            f"expected {preferred!r} or any .ckpt"
        )
    return ckpts[0]


from tempus_bench.models.base_model import BaseModel, validate_inputs, validate_covariate_support


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
        # Train receives covariates aligned with y_context and y_target separately
        # (contiguous history only; not forecast-horizon ``future`` covariates).
        if x_context is None or x_target is None:
            validate_covariate_support(
                x_context,
                x_target,
                supports_past_only=True,
                supports_future_only=False,
                supports_both=False,
                model_name="TabPFN",
            )
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
        # ``predict`` only consumes ``x_context`` (history). Reject true future covariates.
        if x_target is not None:
            raise ValueError(
                "TabPFN does not use x_target during prediction (past covariates only). "
                "Pass the full past covariate block via x_context only."
            )
        validate_covariate_support(
            x_context,
            None,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="TabPFN",
        )
        # Map legacy keys to expected ones for backward compatibility
        context_window = int(kwargs.get("context_window", self.max_sequence_length))
        forecast_window = int(kwargs.get("forecast_window", kwargs.get("prediction_length", self.max_sequence_length)))

        # Determine total horizon from target timestamps
        forecast_horizon = int(getattr(timestamps_target, "shape", [0])[0])

        # Ensure 1D arrays for context/targets
        y_context = np.squeeze(y_context).astype(np.float32)

        # Use last context_window points
        y_hist = y_context[-context_window:]

        # Build time features; extend with x_context (past covariates only) for non-native support
        X_hist = make_time_features(len(y_hist)).values
        has_covariates = x_context is not None
        if has_covariates:
            x_hist = np.asarray(x_context[-len(y_hist):], dtype=np.float32)
            if x_hist.ndim == 1:
                x_hist = x_hist.reshape(-1, 1)
            X_hist = np.concatenate([X_hist, x_hist], axis=1)
            last_cov = x_hist[-1:].astype(np.float32)  # (1, num_covariates)
        model_path: str | Literal["auto"] = "auto"
        hf_or_dir = getattr(self, "hf_model_name", None)
        if hf_or_dir:
            if os.path.isfile(hf_or_dir):
                model_path = hf_or_dir
            elif os.path.isdir(hf_or_dir):
                model_path = _tabpfn_checkpoint_file(hf_or_dir)
        regressor = TabPFNRegressor(model_path=model_path)
        regressor.fit(X_hist, y_hist)

        # Roll out forecasts in chunks
        preds: list[np.ndarray] = []
        remaining = forecast_horizon
        while remaining > 0:
            step = min(forecast_window, remaining)
            # Generate future feature positions immediately following history
            X_future = make_time_features(len(y_hist) + step).values[-step:]
            if has_covariates:
                x_future_pad = np.tile(last_cov, (step, 1))
                X_future = np.concatenate([X_future, x_future_pad], axis=1)
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