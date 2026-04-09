"""
Theta model for univariate, multivariate, and covariate (stacked exogenous) benchmarks.

**Multivariate tasks** (multiple joint targets, no exogenous stack): we fit one
``sktime.forecasting.theta.ThetaForecaster`` per target column. This matches how
other classical models in TempusBench treat multivariate data (e.g. separate
univariate fits per channel) and avoids ill-conditioned joint Θ-matrix estimation
on short windows.

**Covariate tasks**: optional ``past_covariates`` / ``x_context`` are concatenated
as extra columns during the joint Θ-line pipeline; forecasts are returned only for
the primary target width.

**Joint Θ-matrix path** (used for univariate with optional stacked covariates, or
when explicitly not using per-channel independence): see
https://onlinelibrary.wiley.com/doi/full/10.1002/for.2334
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from sklearn.linear_model import LinearRegression
from sktime.forecasting.theta import ThetaForecaster

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ThetaHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    sp: int = Field(..., ge=1, description="Seasonal period")
    theta_method: Literal["least_squares", "correlation_optimal"] = Field(
        ..., description="Method for theta estimation"
    )
    # Fixed Hyperparameters - Optional for User to override
    use_reduced_rank: bool = Field(
        default=False, description="Whether to use cointegration/reduced rank"
    )


def _resolve_exogenous(
    x_context: Optional[np.ndarray], kwargs: dict
) -> Optional[np.ndarray]:
    if x_context is not None:
        return x_context
    pc = kwargs.get("past_covariates")
    if pc is None:
        return None
    if not isinstance(pc, np.ndarray):
        return np.asarray(pc, dtype=float)
    return pc


def _last_finite_scalar_1d(values: np.ndarray) -> float:
    """Last finite value along a 1D array, or 0.0 when none exist."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(finite[-1])


def _validate_covariate_rows(
    y_rows: int, x_arr: np.ndarray, *, label: str
) -> None:
    if x_arr.ndim != 2:
        raise ValueError(
            f"{label} must be 2D (num_steps, num_covariates), got shape {x_arr.shape}"
        )
    if x_arr.shape[0] != y_rows:
        raise ValueError(
            f"{label} num_steps must match y_context: expected {y_rows}, got {x_arr.shape[0]}"
        )


class ThetaModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ThetaHyperparams)

        self._models: list = []
        self._num_original_targets: int = 0
        self._trained_with_covariates: bool = False
        self._independent_multivariate: bool = False
        self._n_covariate_channels: int = 0
        self._num_forecast_targets: int = 0
        self.num_targets: int = 0
        # Per-channel constant fallback for independent multivariate (zero-variance series).
        self._independent_constant_last: list[float] = []
        # Joint path: last detrended theta-line value when ThetaForecaster refuses to fit.
        self._joint_constant_last: list[float] = []
        # y_work has < 2 rows: skip drift + Θ-matrix (same as per-channel univariate Theta).
        self._per_column_theta_only: bool = False

    def _estimate_drift(self, y_context: np.ndarray) -> np.ndarray:
        diff_data = np.diff(y_context, axis=0)
        return np.mean(diff_data, axis=0)

    def _detrend_data(
        self, y_context: np.ndarray, drift_vector: np.ndarray
    ) -> np.ndarray:
        t_n = y_context.shape[0]
        time_index = np.arange(1, t_n + 1).reshape(-1, 1)
        linear_trend = time_index @ drift_vector.reshape(1, -1)
        return y_context - linear_trend

    def _estimate_theta_matrix_least_squares(
        self, detrended_data: np.ndarray, num_targets: int
    ) -> np.ndarray:
        theta_matrix = np.zeros((num_targets, num_targets))
        diff_data = np.diff(detrended_data, axis=0)
        lagged_data = detrended_data[:-1, :]
        for i in range(num_targets):
            y = diff_data[:, i]
            x = lagged_data
            reg = LinearRegression(fit_intercept=False)
            reg.fit(x, y)
            theta_matrix[i, :] = reg.coef_
        return theta_matrix

    def _estimate_theta_matrix_correlation_optimal(
        self, detrended_data: np.ndarray, num_targets: int
    ) -> np.ndarray:
        diff_data = np.diff(detrended_data, axis=0)
        theta_matrix = np.eye(num_targets)
        if diff_data.shape[0] < 2:
            return theta_matrix
        corr_matrix = np.corrcoef(diff_data.T)
        corr_matrix = np.nan_to_num(
            corr_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )
        np.fill_diagonal(corr_matrix, 1.0)
        for i in range(num_targets):
            for j in range(num_targets):
                if i != j:
                    theta_matrix[i, j] = 0.5 * corr_matrix[i, j]
        return theta_matrix

    def _create_theta_lines(
        self, detrended_data: np.ndarray, theta_matrix: np.ndarray
    ) -> np.ndarray:
        return detrended_data @ theta_matrix.T

    def _fit_theta_forecaster_on_series(
        self, values_1d: np.ndarray, *, sp: int
    ) -> Optional[ThetaForecaster]:
        series = pd.Series(np.asarray(values_1d, dtype=float))
        if series.isna().any():
            series = series.interpolate(limit_direction="both")
            series = series.fillna(0.0)
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            raise ValueError("Empty series in ThetaModel._fit_theta_forecaster_on_series")
        std = float(np.nanstd(arr))
        if (not np.isfinite(std)) or std < 1e-15:
            # ThetaForecaster cannot fit all-NaN / strictly constant series; caller uses last-value forecast.
            return None
        has_non_positive = (series <= 0).any()
        insufficient_cycles = len(series) < 2 * sp
        deseasonalize = (not has_non_positive) and (not insufficient_cycles)
        theta_model = ThetaForecaster(sp=sp, deseasonalize=deseasonalize)
        theta_model.fit(y=series)
        return theta_model

    @validate_inputs
    def train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        past_covariates: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> "ThetaModel":
        x_stack = x_context if x_context is not None else past_covariates
        if x_stack is None:
            x_stack = _resolve_exogenous(None, kwargs)

        self._num_original_targets = int(y_context.shape[1])
        self._models = []
        self._per_column_theta_only = False
        self._joint_constant_last = []

        if x_stack is not None:
            x_stack = np.asarray(x_stack, dtype=float)
            _validate_covariate_rows(y_context.shape[0], x_stack, label="past_covariates")
            self._trained_with_covariates = True
            self._n_covariate_channels = x_stack.shape[1]
            self._num_forecast_targets = self._num_original_targets
            self._independent_multivariate = False
            y_work = np.concatenate([y_context, x_stack], axis=1)
        else:
            self._trained_with_covariates = False
            self._n_covariate_channels = 0
            y_work = y_context
            self._num_forecast_targets = self._num_original_targets
            # Several joint targets without exogenous stack: independent univariate Theta per column.
            self._independent_multivariate = y_work.shape[1] > 1

        sp = self.sp
        theta_method = self.theta_method
        num_targets = y_work.shape[1]
        self.num_targets = num_targets

        if self._independent_multivariate:
            self.drift_vector = None
            self.theta_matrix = None
            self._independent_constant_last = []
            for j in range(self._num_forecast_targets):
                col = y_work[:, j]
                fitted = self._fit_theta_forecaster_on_series(col, sp=sp)
                if fitted is None:
                    self._models.append(None)
                    self._independent_constant_last.append(
                        _last_finite_scalar_1d(np.asarray(col, dtype=float))
                    )
                else:
                    self._models.append(fitted)
                    self._independent_constant_last.append(float("nan"))
            self.is_fitted = True
            return self

        # Need ≥2 rows for diff-based drift and Θ estimation (joint path).
        if y_work.shape[0] < 2:
            self._per_column_theta_only = True
            self.drift_vector = None
            self.theta_matrix = None
            self._independent_constant_last = []
            for j in range(num_targets):
                col = y_work[:, j]
                fitted = self._fit_theta_forecaster_on_series(col, sp=sp)
                if fitted is None:
                    self._models.append(None)
                    self._independent_constant_last.append(
                        _last_finite_scalar_1d(np.asarray(col, dtype=float))
                    )
                else:
                    self._models.append(fitted)
                    self._independent_constant_last.append(float("nan"))
            self.is_fitted = True
            return self

        self.drift_vector = np.nan_to_num(
            self._estimate_drift(y_work), nan=0.0, posinf=0.0, neginf=0.0
        )
        detrended_data = self._detrend_data(y_work, self.drift_vector)

        if theta_method == "correlation_optimal":
            self.theta_matrix = self._estimate_theta_matrix_correlation_optimal(
                detrended_data, num_targets
            )
        elif theta_method == "least_squares":
            self.theta_matrix = self._estimate_theta_matrix_least_squares(
                detrended_data, num_targets
            )
        else:
            raise ValueError(f"Unknown theta_method: {theta_method!r}")

        theta_lines = self._create_theta_lines(detrended_data, self.theta_matrix)
        self._joint_constant_last = []
        for i in range(num_targets):
            col = theta_lines[:, i]
            fitted = self._fit_theta_forecaster_on_series(col, sp=sp)
            if fitted is None:
                self._models.append(None)
                self._joint_constant_last.append(
                    _last_finite_scalar_1d(np.asarray(col, dtype=float))
                )
            else:
                self._models.append(fitted)
                self._joint_constant_last.append(float("nan"))

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
        past_covariates: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")

        x_stack = x_context if x_context is not None else past_covariates
        if x_stack is None:
            x_stack = _resolve_exogenous(None, kwargs)

        if self._trained_with_covariates:
            if x_stack is None:
                raise ValueError(
                    "past_covariates is required at predict when the model was trained with covariates"
                )
        elif x_stack is not None:
            raise ValueError(
                "past_covariates were provided at predict but the model was trained without covariates"
            )

        forecast_horizon = int(timestamps_target.shape[0])
        fh = np.arange(1, forecast_horizon + 1)

        if self._independent_multivariate:
            k = self._num_forecast_targets
            out = np.zeros((forecast_horizon, k), dtype=float)
            for j in range(k):
                m = self._models[j]
                if m is None:
                    cval = self._independent_constant_last[j]
                    out[:, j] = cval
                else:
                    pred = m.predict(fh=fh)
                    vals = np.asarray(pred, dtype=float).ravel()
                    if vals.size < forecast_horizon:
                        tail = (
                            float(vals[-1])
                            if vals.size > 0
                            else self._independent_constant_last[j]
                        )
                        pad = np.full(forecast_horizon - vals.size, tail, dtype=float)
                        vals = np.concatenate([vals, pad])
                    out[:, j] = vals[:forecast_horizon]
            return out

        if self._per_column_theta_only:
            k = len(self._models)
            out = np.zeros((forecast_horizon, k), dtype=float)
            for j in range(k):
                m = self._models[j]
                if m is None:
                    cval = self._independent_constant_last[j]
                    out[:, j] = cval
                else:
                    pred = m.predict(fh=fh)
                    vals = np.asarray(pred, dtype=float).ravel()
                    if vals.size < forecast_horizon:
                        tail = (
                            float(vals[-1])
                            if vals.size > 0
                            else self._independent_constant_last[j]
                        )
                        pad = np.full(forecast_horizon - vals.size, tail, dtype=float)
                        vals = np.concatenate([vals, pad])
                    out[:, j] = vals[:forecast_horizon]
            return out[:, : self._num_original_targets]

        if x_stack is not None:
            x_stack = np.asarray(x_stack, dtype=float)
            _validate_covariate_rows(
                y_context.shape[0], x_stack, label="past_covariates"
            )
            y_w = np.concatenate([y_context, x_stack], axis=1)
        else:
            y_w = y_context

        num_targets = y_w.shape[1]
        num_original_targets = self._num_original_targets or y_context.shape[1]
        all_predictions = np.zeros((forecast_horizon, num_targets), dtype=float)
        future_times = np.arange(1, forecast_horizon + 1, dtype=float)
        for i in range(num_targets):
            drift_i = float(
                np.nan_to_num(
                    self.drift_vector[i], nan=0.0, posinf=0.0, neginf=0.0
                )
            )
            linear_trend = future_times * drift_i
            mdl = self._models[i]
            if mdl is None:
                base = self._joint_constant_last[i]
                all_predictions[:, i] = base + linear_trend
                continue
            theta_forecast = mdl.predict(fh=fh)
            vals = np.asarray(theta_forecast, dtype=float).ravel()
            if vals.size < forecast_horizon:
                tail = (
                    float(vals[-1])
                    if vals.size > 0
                    else self._joint_constant_last[i]
                )
                pad = np.full(forecast_horizon - vals.size, tail, dtype=float)
                vals = np.concatenate([vals, pad])
            all_predictions[:, i] = vals[:forecast_horizon] + linear_trend
        return all_predictions[:, :num_original_targets]
