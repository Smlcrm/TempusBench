"""
Chronax Theta (Standard Theta Method) model implementation.

This module provides a TempusBench wrapper around ``chronax.models.Theta``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax Theta is fit for each variate.
    * Chronax Theta declares ``uses_exog = False``; exogenous covariates are
      accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import Theta

from tempus_bench.models.base_model import BaseModel, validate_inputs


DecompLiteral = Literal["multiplicative", "additive"]


class ChronaxThetaHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.Theta`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    season_length: int = Field(..., ge=1, description="Seasonal period length")
    decomposition_type: DecompLiteral = Field(..., description="Decomposition type")
    n_samples: int = Field(..., ge=1, description="Number of sample paths for intervals")


ChronaxThetaHyperparams.model_rebuild()


class ChronaxThetaModel(BaseModel):
    """
    Chronax Theta model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax Theta is fit
    for each target. The fitted instances are stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxThetaHyperparams)
        self._models: list[Theta] = []

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
    ) -> Theta:
        """Fit a single Chronax Theta on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        model = Theta(
            season_length=int(self.season_length),
            decomposition_type=str(self.decomposition_type),
            n_samples=int(self.n_samples),
        )
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        theta_model: Theta,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax Theta."""
        if not self.is_fitted:
            raise ValueError("ChronaxThetaModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax Theta prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = theta_model.predict(h=forecast_steps)
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
    ) -> "ChronaxThetaModel":
        """Train a separate Chronax Theta for each variate."""
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
            raise ValueError("ChronaxThetaModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                theta_model=fitted_model,
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


class ThetaChronaxModel(ChronaxThetaModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
