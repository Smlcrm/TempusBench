"""
Chronax ETS (Error/Trend/Seasonal exponential smoothing) model implementation.

This module provides a TempusBench wrapper around ``chronax.models.ETS``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax ETS is fit for each variate.
    * Chronax ETS declares ``uses_exog = False``; exogenous covariates are
      accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import ETS

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxEtsHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.ETS`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    season_length: int = Field(..., ge=1, description="Seasonal period length")
    model: str = Field(..., min_length=3, max_length=3, description="ETS model string (e.g. 'ANN')")
    damped: bool | None = Field(..., description="Whether to damp the trend")
    phi: float | None = Field(..., description="Damping parameter")
    max_iter: int | None = Field(..., description="Maximum optimization iterations")
    optax_lr: float = Field(..., gt=0, description="Optax learning rate")
    optax_clip: float = Field(..., gt=0, description="Optax gradient clipping")


ChronaxEtsHyperparams.model_rebuild()


class ChronaxEtsModel(BaseModel):
    """
    Chronax ETS model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax ETS is fit for
    each target. The fitted instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxEtsHyperparams)
        self._models: list[ETS] = []

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
    ) -> ETS:
        """Fit a single Chronax ETS on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        def _build(
            season_length: int,
            model_str: str,
            damped: bool | None,
            phi: float | None,
        ) -> ETS:
            return ETS(
                season_length=season_length,
                model=model_str,
                damped=damped,
                phi=phi,
                max_iter=self.max_iter,
                optax_lr=float(self.optax_lr),
                optax_clip=float(self.optax_clip),
                prediction_intervals=None,
            )

        model_str = str(self.model).upper()
        season_len = int(self.season_length)
        model = _build(season_len, model_str, self.damped, self.phi)
        try:
            model.fit(endog)
        except ValueError as exc:
            msg = str(exc).lower()
            if "admissible" not in msg:
                raise
            # Non-seasonal ANN with no damping is always admissible; bad combos
            # (e.g. damped trend on ANN) raise from Chronax's ETS spec checker.
            fallback = _build(1, "ANN", False, None)
            fallback.fit(endog)
            return fallback
        return model

    @validate_inputs
    def _predict(
        self,
        ets_model: ETS,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax ETS."""
        if not self.is_fitted:
            raise ValueError("ChronaxEtsModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax ETS prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = ets_model.predict(h=forecast_steps)
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
    ) -> "ChronaxEtsModel":
        """Train a separate Chronax ETS for each variate."""
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
            raise ValueError("ChronaxEtsModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                ets_model=fitted_model,
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


class EtsChronaxModel(ChronaxEtsModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
