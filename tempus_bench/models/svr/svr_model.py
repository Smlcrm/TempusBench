"""
SVR model implementation.
"""

import os
import pickle

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tempus_bench.models.base_model import BaseModel, validate_inputs


class SvrHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field(
        ..., description="SVR kernel type"
    )
    C: float = Field(..., gt=0, description="Regularization parameter")
    # Fixed Hyperparameters - Optional for User to override
    epsilon: float = Field(
        default=0.1, ge=0, description="Epsilon parameter for epsilon-SVR"
    )
    gamma: Literal["scale", "auto"] = Field(
        default="scale",
        description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'",
    )


class SvrModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        """
        Initialize Support Vector Regression (SVR) model with model-specific parameters.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.
        """
        super().__init__(params, settings, SvrHyperparams)
        self._scaler = StandardScaler()  # SVR is sensitive to feature scaling
        self._effective_lookback: int = int(self.lookback_window)

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "SvrModel":
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        # Calculate forecast_horizon from y_target if not provided in kwargs
        forecast_horizon = int(kwargs.get("forecast_horizon", y_target.shape[0]))
        lookback_window = int(self.lookback_window)

        if not self.is_fitted:
            self._build_model()

        # Combine context and target for full training series if y_target is provided
        # Handle (num_steps, num_targets) format
        y_actual = np.concatenate([y_context, y_target], axis=0)
        # Zero out the target portion in the X-side series to prevent future y leakage
        y_masked = np.concatenate([y_context, np.zeros_like(y_target)], axis=0)

        # Build combined covariate series (past + future) if provided
        x_series = None
        if x_context is not None and x_target is not None:
            x_series = np.concatenate([x_context, x_target], axis=0)

        self._effective_lookback = self._resolve_effective_lookback(
            series_length=y_actual.shape[0],
            forecast_horizon=forecast_horizon,
            configured_lookback=lookback_window,
        )

        X, y = self._create_features_targets(
            y_masked,
            y_actual,
            forecast_horizon=forecast_horizon,
            lookback_window=self._effective_lookback,
            x_series=x_series,
        )

        # Scale features (SVR is sensitive to feature scaling)
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        self._model.fit(X_scaled, y)
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
        **kwargs: dict,
    ):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        # Calculate forecast_horizon from timestamps_target if not provided in kwargs
        forecast_horizon = int(
            kwargs.get("forecast_horizon", timestamps_target.shape[0])
        )

        # Reference params, settings, device, python_version
        lookback_window = int(self._effective_lookback)

        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        if y_context.shape[0] < lookback_window:
            raise ValueError(
                f"y_context too short: {y_context.shape[0]} < effective lookback {lookback_window}"
            )

        total_steps = len(timestamps_target)
        num_targets = y_context.shape[1]  # num_targets is second dimension

        has_cov = x_context is not None and x_target is not None
        if has_cov:
            x_all = np.concatenate([x_context, x_target], axis=0)
            # Start offset so that x_all[x_offset : x_offset+lookback] aligns with y_context[-lookback:]
            x_offset = len(x_context) - lookback_window

        preds = []
        context = y_context.copy()

        steps_done = 0

        while steps_done < total_steps:
            # Use last effective lookback timesteps (matches training feature width)
            current_window = context[-lookback_window:, :]

            # Build feature vector: past x + future x first, past y last
            feature_parts = []
            if has_cov:
                past_x = x_all[x_offset : x_offset + lookback_window, :]
                future_x = x_all[x_offset + lookback_window : x_offset + lookback_window + forecast_horizon, :]
                feature_parts.append(past_x.flatten())
                feature_parts.append(future_x.flatten())
            feature_parts.append(current_window.flatten())
            y_flat = np.expand_dims(np.concatenate(feature_parts), axis=0)
            # Scale features before prediction (SVR is sensitive to feature scaling)
            y_flat_scaled = self._scaler.transform(y_flat)
            pred = self._model.predict(y_flat_scaled)

            # MultiOutputRegressor returns shape (1, forecast_horizon * num_targets)
            # Flatten to 1D, then reshape to (forecast_horizon, num_targets)
            pred = pred.flatten()
            pred = np.reshape(pred, (forecast_horizon, num_targets))

            # Calculate how many steps we actually need for this iteration
            steps_needed = min(forecast_horizon, total_steps - steps_done)
            pred_steps = pred[:steps_needed, :]  # Only take what we need

            preds.append(pred_steps)

            # Concatenate along time axis (axis=0)
            context = np.concatenate([context, pred_steps], axis=0)
            if has_cov:
                x_offset += steps_needed
            steps_done += steps_needed

        # Concatenate all predictions along time axis
        preds = np.concatenate(preds, axis=0)

        # Return (total_steps, num_targets) - should be exact now
        # Note: Do not inverse-transform targets with feature scaler; predictions are in original target scale
        return preds

    def _build_model(self):
        """
        Build the SVR model instance from the configuration using MultiOutputRegressor for direct multi-output forecasting.
        """
        # Build base SVR from params fields relevant to estimator
        base_svr = SVR(
            kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma
        )
        self._model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    @staticmethod
    def _resolve_effective_lookback(
        *,
        series_length: int,
        forecast_horizon: int,
        configured_lookback: int,
    ) -> int:
        """
        Cap lookback so that a short context+train segment (e.g. covariate GDP tasks,
        small context_window tasks) still yields at least one supervised sample.
        """
        if series_length <= forecast_horizon:
            raise ValueError(
                "Not enough data for SVR: concatenated context+train length "
                f"{series_length} must be greater than forecast_horizon {forecast_horizon}"
            )
        max_feasible = series_length - forecast_horizon
        return max(1, min(int(configured_lookback), max_feasible))

    def _create_features_targets(
        self,
        y_masked: np.ndarray,
        y_actual: np.ndarray,
        forecast_horizon: int,
        lookback_window: int,
        x_series: Optional[np.ndarray] = None,
    ):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.

        X is built from y_masked (future y zeroed out) plus past and future covariate values.
        y is built from y_actual (true future target values).
        """
        X, y = [], []

        num_steps, num_targets = y_masked.shape

        # Validate data length
        min_required_length = lookback_window + forecast_horizon
        if num_steps < min_required_length:
            raise ValueError(
                f"Not enough data for SVR. Have {num_steps} observations, need at least {min_required_length}"
            )

        for i in range(num_steps - lookback_window - forecast_horizon + 1):
            # Past x and future x first, then past y last
            feature_parts = []
            if x_series is not None:
                feature_parts.append(x_series[i : i + lookback_window, :].flatten())
                feature_parts.append(x_series[i + lookback_window : i + lookback_window + forecast_horizon, :].flatten())
            feature_parts.append(y_masked[i : i + lookback_window, :].flatten())

            X.append(np.concatenate(feature_parts))

            # Labels: actual future y values only
            curr_y = y_actual[
                i + lookback_window : i + lookback_window + forecast_horizon, :
            ].flatten()
            y.append(curr_y)

        X, y = np.array(X), np.array(y)

        return X, y
