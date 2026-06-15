"""
Chronax RandomWalkWithDrift model implementation.

This module provides a TempusBench wrapper around ``chronax.models.RandomWalkWithDrift``.
Mirrors the conventions of ``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax RandomWalkWithDrift is fit for each variate.
    * Chronax RandomWalkWithDrift declares ``uses_exog = False`` and its
      ``predict(h, level=None)`` signature does not accept ``X``. Exogenous
      covariates are accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

from chronax.models import RandomWalkWithDrift

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxRandomWalkWithDriftHyperparams(PydanticBaseModel):
    """Pydantic schema for Chronax RandomWalkWithDrift (no required hyperparameters)."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


ChronaxRandomWalkWithDriftHyperparams.model_rebuild()


class ChronaxRandomWalkWithDriftModel(BaseModel):
    """
    Chronax RandomWalkWithDrift model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax
    RandomWalkWithDrift is fit for each target. The fitted instances are stored
    in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxRandomWalkWithDriftHyperparams)
        self._models: list[RandomWalkWithDrift] = []

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
    ) -> RandomWalkWithDrift:
        """Fit a single Chronax RandomWalkWithDrift on one variate of the context data."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float64)
        model = RandomWalkWithDrift()
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        random_walk_with_drift_model: RandomWalkWithDrift,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax RandomWalkWithDrift."""
        if not self.is_fitted:
            raise ValueError("ChronaxRandomWalkWithDriftModel not fitted. Call train() first.")

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax RandomWalkWithDrift prediction."
            )

        forecast_steps = int(len(timestamps_target))
        # Chronax's RandomWalkWithDrift.predict has signature (h, level=None); it does not accept X.
        output = random_walk_with_drift_model.predict(h=forecast_steps)
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
    ) -> "ChronaxRandomWalkWithDriftModel":
        """Train a separate Chronax RandomWalkWithDrift for each variate."""
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
            raise ValueError("ChronaxRandomWalkWithDriftModel not fitted. Call train() first.")

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                random_walk_with_drift_model=fitted_model,
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


class RandomWalkWithDriftChronaxModel(ChronaxRandomWalkWithDriftModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
