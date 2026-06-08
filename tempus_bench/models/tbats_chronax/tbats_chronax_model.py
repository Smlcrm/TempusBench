"""
Chronax TBATS (fixed-configuration Trigonometric-Box-Cox-ARMA-Trend-Seasonal) model.

This module provides a TempusBench wrapper around ``chronax.models.TBATS``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax TBATS is fit for each variate.
    * Chronax TBATS declares ``uses_exog = False``; exogenous covariates are
      accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import TBATS

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxTbatsHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.TBATS`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    season_length: int | list[int] = Field(..., description="Seasonal period(s); int or list of ints")
    use_boxcox: bool | None = Field(..., description="Apply Box-Cox transformation")
    use_trend: bool | None = Field(..., description="Include trend component")
    use_damped_trend: bool | None = Field(..., description="Damp trend component")
    use_arma_errors: bool = Field(..., description="Use ARMA residual structure")
    bc_lower_bound: float = Field(default=-1.0, description="Box-Cox lower bound (not tuned)")
    bc_upper_bound: float = Field(default=2.0, description="Box-Cox upper bound (not tuned)")


ChronaxTbatsHyperparams.model_rebuild()


def _normalize_season_length(sl: int | list[int]) -> int | list[int]:
    """Pass through Chronax's int|list[int] contract, coercing values to ints."""
    if isinstance(sl, list):
        return [int(s) for s in sl]
    return int(sl)


class ChronaxTbatsModel(BaseModel):
    """
    Chronax TBATS model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax TBATS is fit
    for each target. The fitted instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxTbatsHyperparams)
        self._models: list[TBATS] = []

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
    ) -> TBATS:
        """Fit a single Chronax TBATS on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        model = TBATS(
            season_length=_normalize_season_length(self.season_length),
            use_boxcox=self.use_boxcox,
            bc_lower_bound=float(self.bc_lower_bound),
            bc_upper_bound=float(self.bc_upper_bound),
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=bool(self.use_arma_errors),
        )
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        tbats_model: TBATS,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax TBATS."""
        if not self.is_fitted:
            raise ValueError("ChronaxTbatsModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax TBATS prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = tbats_model.predict(h=forecast_steps)
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
    ) -> "ChronaxTbatsModel":
        """Train a separate Chronax TBATS for each variate."""
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
            raise ValueError("ChronaxTbatsModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                tbats_model=fitted_model,
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


class TbatsChronaxModel(ChronaxTbatsModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
