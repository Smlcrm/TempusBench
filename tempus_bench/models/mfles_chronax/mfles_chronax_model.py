"""
Chronax MFLES (Multi-Feature Locally Exponential Smoothing) model implementation.

This module provides a TempusBench wrapper around ``chronax.models.MFLES``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax MFLES is fit for each variate.
    * Optional past exogenous regressors (``x_context``) passed to
      ``MFLES.fit(y, X=...)``.
    * Optional future exogenous regressors (``x_target``) passed to
      ``MFLES.predict(h, X=...)``.

Chronax's ``MFLES`` declares ``uses_exog = True`` and accepts ``X`` in both
``fit`` and ``predict``. Unlike most Chronax forecasters, MFLES takes
model-shape hyperparameters in ``fit(...)`` rather than ``__init__`` — only
``verbose`` and ``robust`` are constructor arguments.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import MFLES

from tempus_bench.models.base_model import BaseModel, validate_inputs


SesMode = Literal["off", "lite", "full", "adaptive"]


class ChronaxMflesHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.MFLES`` constructor + fit kwargs."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    verbose: int = Field(..., ge=0, description="Verbosity level")
    robust: bool | None = Field(..., description="Siegel regression for trend; None auto-detects")
    seasonal_period: int | list[int] | None = Field(default=None, description="Seasonal period(s) (not tuned)")
    fourier_order: int | None = Field(default=None, description="Fourier order; None auto-selects (not tuned)")
    ma: int | list[int] | None = Field(default=None, description="Moving-average window(s) for residual smoothing (not tuned)")
    alpha: float = Field(default=1.0, description="LASSO regularization strength for changepoint trend (not tuned)")
    decay: float = Field(default=-1.0, description="Legacy; unused but retained for API compatibility (not tuned)")
    n_changepoints: float | int = Field(default=0.25, description="Changepoint count; float<1 -> fraction of n (not tuned)")
    seasonal_lr: float = Field(default=0.9, description="Seasonal update learning rate (not tuned)")
    rs_lr: float = Field(default=1.0, description="Residual-smoothing learning rate (not tuned)")
    linear_lr: float = Field(default=0.9, description="Trend update learning rate (not tuned)")
    cov_threshold: float = Field(default=0.7, description="CoV threshold for auto robust-mode detection (not tuned)")
    moving_medians: bool = Field(default=False, description="Use period-wise medians to initialize fitted values (not tuned)")
    max_rounds: int = Field(default=50, ge=1, description="Maximum fitting iterations (not tuned)")
    min_alpha: float = Field(default=0.05, description="Minimum SES alpha in ensemble grid (not tuned)")
    max_alpha: float = Field(default=1.0, description="Maximum SES alpha in ensemble grid (not tuned)")
    round_penalty: float = Field(default=0.0001, description="Improvement threshold for accepting residual-smoothing update (not tuned)")
    trend_penalty: bool = Field(default=True, description="Dampen trend slope by R-squared penalty (not tuned)")
    multiplicative: bool | None = Field(default=None, description="Fit in log-space; None auto-detects (not tuned)")
    changepoints: bool = Field(default=True, description="Enable piecewise linear trend via LASSO (not tuned)")
    smoother: bool = Field(default=False, description="Only used when ses_mode='adaptive' (not tuned)")
    ses_mode: SesMode = Field(default="lite", description="Residual smoothing strategy (not tuned)")
    seasonality_weights: bool = Field(default=False, description="Recency-weighted OLS for seasonal fitting (not tuned)")
    gradient_strategy: bool = Field(default=False, description="Legacy; unused (not tuned)")


ChronaxMflesHyperparams.model_rebuild()


def _to_jax_array(x: Optional[np.ndarray]) -> Optional[jnp.ndarray]:
    """Convert an optional NumPy array to a float64 JAX array, preserving ``None``."""
    if x is None:
        return None
    return jnp.asarray(x, dtype=jnp.float64)


def _normalize_period(sp: int | list[int] | None) -> int | list[int] | None:
    if sp is None:
        return None
    if isinstance(sp, list):
        return [int(s) for s in sp]
    return int(sp)


def _normalize_ma(ma: int | list[int] | None) -> int | list[int] | None:
    if ma is None:
        return None
    if isinstance(ma, list):
        return [int(v) for v in ma]
    return int(ma)


class ChronaxMflesModel(BaseModel):
    """
    Chronax MFLES model for univariate/multivariate time series forecasting with
    optional past/future exogenous regressors.

    Supports both standard MFLES and multi-seasonal MFLES via ``seasonal_period``
    as a list. Exogenous regressors for both the context (past) and target
    (future) horizons are forwarded to Chronax when provided.

    The fitted per-target Chronax MFLES instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxMflesHyperparams)
        self._models: list[MFLES] = []

    @validate_inputs
    def _train(
        self,
        y_context: np.ndarray,
        y_target: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> MFLES:
        """Fit a single Chronax MFLES on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        exog = _to_jax_array(x_context)

        model = MFLES(verbose=int(self.verbose), robust=self.robust)
        model.fit(
            endog,
            seasonal_period=_normalize_period(self.seasonal_period),
            X=exog,
            fourier_order=self.fourier_order,
            ma=_normalize_ma(self.ma),
            alpha=float(self.alpha),
            decay=float(self.decay),
            n_changepoints=self.n_changepoints,
            seasonal_lr=float(self.seasonal_lr),
            rs_lr=float(self.rs_lr),
            exogenous_lr=1.0,
            exogenous_estimator=None,
            exogenous_params={},
            linear_lr=float(self.linear_lr),
            cov_threshold=float(self.cov_threshold),
            moving_medians=bool(self.moving_medians),
            max_rounds=int(self.max_rounds),
            min_alpha=float(self.min_alpha),
            max_alpha=float(self.max_alpha),
            round_penalty=float(self.round_penalty),
            trend_penalty=bool(self.trend_penalty),
            multiplicative=self.multiplicative,
            changepoints=bool(self.changepoints),
            smoother=bool(self.smoother),
            ses_mode=str(self.ses_mode),
            seasonality_weights=bool(self.seasonality_weights),
            gradient_strategy=bool(self.gradient_strategy),
        )
        return model

    @validate_inputs
    def _predict(
        self,
        mfles_model: MFLES,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax MFLES."""
        if not self.is_fitted:
            raise ValueError("ChronaxMflesModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax MFLES prediction."
            )

        forecast_steps = int(len(timestamps_target))
        use_covariates = bool(kwargs.pop("use_covariates", True))
        exog_future = _to_jax_array(x_target) if use_covariates else None
        output = mfles_model.predict(h=forecast_steps, X=exog_future)
        y_pred = np.asarray(output["mean"], dtype=np.float64).reshape(-1, 1)
        return y_pred

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
    ) -> "ChronaxMflesModel":
        """Train a separate Chronax MFLES for each variate."""
        num_targets = y_context.shape[1]
        self._models = []

        for k in range(num_targets):
            fitted_model = self._train(
                y_context=y_context[:, k : k + 1],
                y_target=y_target[:, k : k + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                x_context=x_context,
                x_target=x_target,
                **kwargs,
            )
            self._models.append(fitted_model)

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
    ) -> np.ndarray:
        """Predict future values for each variate and concatenate the results."""
        if not self.is_fitted:
            raise ValueError("ChronaxMflesModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                mfles_model=fitted_model,
                y_context=y_context[:, idx : idx + 1],
                timestamps_context=timestamps_context,
                timestamps_target=timestamps_target,
                x_context=x_context,
                x_target=x_target,
                **kwargs,
            )
            preds.append(prediction)

        result = np.concatenate(preds, axis=-1)
        return result


class MflesChronaxModel(ChronaxMflesModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
