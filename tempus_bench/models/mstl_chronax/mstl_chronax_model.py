"""
Chronax MSTL (Multiple Seasonal-Trend decomposition using LOESS) model.

This module provides a TempusBench wrapper around ``chronax.models.MSTL``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax MSTL is fit for each variate.
    * Chronax MSTL declares ``uses_exog = False``; exogenous covariates are
      accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, model_validator

from chronax.models import MSTL

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxMstlHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.MSTL`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    period: int | list[int] = Field(..., description="Seasonal period(s); int or list of ints")
    iterate: int = Field(..., ge=1, description="Number of STL iterations")
    seasonal_deg: int = Field(..., ge=0, le=1, description="Degree of seasonal LOESS")
    trend_deg: int = Field(..., ge=0, le=1, description="Degree of trend LOESS")
    seasonal_jump: int = Field(..., ge=1, description="Seasonal LOESS jump size")
    trend_jump: int = Field(..., ge=1, description="Trend LOESS jump size")
    inner: int = Field(..., ge=1, description="Inner STL iterations")
    fitted: bool = Field(..., description="Whether to retain fitted values")
    s_window: int | list[int] | None = Field(
        default=None,
        description="Seasonal smoothing window(s); None uses Chronax default (not tuned)",
    )
    trend_window: int | None = Field(default=None, description="Trend LOESS window length (not tuned)")
    low_pass: int | None = Field(default=None, description="Low-pass filter window length (not tuned)")
    tail_window: int | None = Field(default=None, description="Tail window for forecasting (not tuned)")

    @model_validator(mode="after")
    def _match_windows(self) -> "ChronaxMstlHyperparams":
        if self.s_window is None:
            return self
        period_list = self.period if isinstance(self.period, list) else [self.period]
        sw_list = self.s_window if isinstance(self.s_window, list) else [self.s_window]
        if len(sw_list) != len(period_list):
            raise ValueError("s_window must have the same length as period.")
        return self


ChronaxMstlHyperparams.model_rebuild()


class ChronaxMstlModel(BaseModel):
    """
    Chronax MSTL model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax MSTL is fit
    for each target. The fitted instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxMstlHyperparams)
        self._models: list[MSTL] = []

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
    ) -> MSTL:
        """Fit a single Chronax MSTL on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)

        if isinstance(self.period, list):
            period_arg: int | list[int] = [int(p) for p in self.period]
        else:
            period_arg = int(self.period)

        s_window_arg: int | list[int] | None
        if self.s_window is None:
            s_window_arg = None
        elif isinstance(self.s_window, list):
            s_window_arg = [int(w) for w in self.s_window]
        else:
            s_window_arg = int(self.s_window)

        model = MSTL(
            period=period_arg,
            iterate=int(self.iterate),
            s_window=s_window_arg,
            seasonal_deg=int(self.seasonal_deg),
            trend_deg=int(self.trend_deg),
            seasonal_jump=int(self.seasonal_jump),
            trend_jump=int(self.trend_jump),
            inner=int(self.inner),
            trend_window=self.trend_window,
            low_pass=self.low_pass,
            tail_window=self.tail_window,
            fitted=bool(self.fitted),
        )
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        mstl_model: MSTL,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax MSTL."""
        if not self.is_fitted:
            raise ValueError("ChronaxMstlModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax MSTL prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = mstl_model.predict(h=forecast_steps)
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
    ) -> "ChronaxMstlModel":
        """Train a separate Chronax MSTL for each variate."""
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
            raise ValueError("ChronaxMstlModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                mstl_model=fitted_model,
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


class MstlChronaxModel(ChronaxMstlModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
