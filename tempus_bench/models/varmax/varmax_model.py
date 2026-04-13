"""
Multivariate VARMAX model.
"""

import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field
from statsmodels.tsa.statespace.varmax import VARMAX

from tempus_bench.models.base_model import BaseModel, validate_inputs

warnings.filterwarnings("ignore")


class VarmaxHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    p: int = Field(..., ge=0, description="Number of AR parameters")
    q: int = Field(..., ge=0, description="Number of MA parameters")
    # Fixed Hyperparameters - Optional for User to override
    trend: Literal["n", "c", "t", "ct"] = Field(
        default="c",
        description="Deterministic trend: 'n' none, 'c' constant, 't' linear, 'ct' both",
    )


def build_varmax_endog(
    y: np.ndarray,
    x: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Stack targets and covariates into a multivariate endogenous matrix for VARMAX.

    When ``x`` is provided, each column of ``x`` is treated as an additional endogenous
    variate (joint VAR), so a single target column with at least one covariate column
    yields a valid multivariate system.

    Returns:
        ``(endog, num_target_columns)`` where forecasts should retain only the first
        ``num_target_columns`` columns for evaluation against ``y_true``.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if x is None:
        if y_arr.shape[1] < 2:
            raise ValueError(
                "VARMAX requires at least two endogenous targets when no covariates are stacked; "
                f"got y.shape={y_arr.shape}. Add covariates or use a multivariate target task."
            )
        return y_arr, y_arr.shape[1]

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            "Covariate row count must match target row count for VARMAX stacking: "
            f"y rows={y_arr.shape[0]}, x rows={x_arr.shape[0]}"
        )
    endog = np.hstack([y_arr, x_arr])
    return endog, y_arr.shape[1]


class VarmaxModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, VarmaxHyperparams)
        self._n_target_cols: int = 0

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
    ) -> "VarmaxModel":
        """Train VARMAX on context + train targets (and optional stacked covariates)."""
        y_ctx = np.asarray(y_context, dtype=float)
        if y_ctx.ndim != 2:
            raise ValueError(f"y_context must be 2D, got shape {y_context.shape}")

        ts_tgt = np.asarray(y_target, dtype=float) if y_target is not None else None
        if ts_tgt is not None and ts_tgt.size > 0:
            y_all = np.vstack([y_ctx, ts_tgt])
        else:
            y_all = y_ctx

        x_all: Optional[np.ndarray] = None
        if x_context is not None:
            x0 = np.asarray(x_context, dtype=float)
            if x0.ndim == 1:
                x0 = x0.reshape(-1, 1)
            if x_target is not None and np.asarray(x_target).size > 0:
                x1 = np.asarray(x_target, dtype=float)
                if x1.ndim == 1:
                    x1 = x1.reshape(-1, 1)
                x_all = np.vstack([x0, x1])
            else:
                if x0.shape[0] != y_all.shape[0]:
                    raise ValueError(
                        "VARMAX with covariates requires x_context with one row per target row "
                        f"(context+train). Got x_context.shape[0]={x0.shape[0]}, "
                        f"y_all.shape[0]={y_all.shape[0]}"
                    )
                x_all = x0

        endog, n_targets = build_varmax_endog(y_all, x_all)
        self._has_exog = x_context is not None
        self._n_target_cols = n_targets

        p = int(self.p)
        q = int(self.q)
        trend = self.trend

        self._convert_to_datetimeindex(timestamps_context)
        if not self.is_fitted:
            # Non-stationary / ill-conditioned panels often break the default stationary
            # state initialization (Schur / PD failures). Approximate diffuse is far more
            # robust on benchmark tasks while remaining a standard statespace choice.
            model = VARMAX(
                endog,
                order=(p, q),
                trend=trend,
                exog=None,
                initialization="approximate_diffuse",
            )
            self._model = model.fit(disp=False, maxiter=500)

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
        """Forecast using the fitted VARMAX model (first ``num_targets`` columns only)."""
        if self._model is None:
            raise ValueError("Model not fitted. Call train first.")

        forecast_steps = int(np.asarray(timestamps_target).shape[0])
        del y_context, timestamps_context, x_context, x_target

        forecasts = self._model.forecast(steps=forecast_steps, exog=None)
        out = np.asarray(forecasts)
        k = self._n_target_cols
        if out.shape[1] < k:
            raise ValueError(
                f"VARMAX forecast has {out.shape[1]} columns but expected at least {k}"
            )
        return out[:, :k]

    def _convert_to_datetimeindex(self, timestamps):
        timestamps = np.squeeze(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            if isinstance(timestamps[0], (int, np.integer)):
                if timestamps[0] > 1e18:
                    timestamps = pd.to_datetime(timestamps, unit="ns")
                elif timestamps[0] > 1e15:
                    timestamps = pd.to_datetime(timestamps, unit="us")
                elif timestamps[0] > 1e12:
                    timestamps = pd.to_datetime(timestamps, unit="ms")
                else:
                    timestamps = pd.to_datetime(timestamps, unit="s")
            else:
                timestamps = pd.to_datetime(timestamps)
        return timestamps
