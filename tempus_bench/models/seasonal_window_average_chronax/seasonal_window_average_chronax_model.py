"""
Chronax SeasonalWindowAverage model implementation.

This module provides a TempusBench wrapper around
``chronax.models.SeasonalWindowAverage``. Mirrors the conventions of
``tempus_bench/models/arima/arima_model.py``:

    * Per-target looping for (potentially) multivariate series — a separate
      Chronax SeasonalWindowAverage is fit for each variate.
    * Chronax SeasonalWindowAverage declares ``uses_exog = False``; exogenous
      covariates are accepted for API consistency but are not forwarded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

from chronax.models import SeasonalWindowAverage

from tempus_bench.models.base_model import BaseModel, validate_inputs


class ChronaxSeasonalWindowAverageHyperparams(PydanticBaseModel):
    """Pydantic schema mirroring the ``chronax.models.SeasonalWindowAverage`` constructor."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    season_length: int = Field(..., ge=2, description="Seasonal period length")
    window_size: int = Field(..., ge=1, description="Number of past seasons averaged")


ChronaxSeasonalWindowAverageHyperparams.model_rebuild()


class ChronaxSeasonalWindowAverageModel(BaseModel):
    """
    Chronax SeasonalWindowAverage model for univariate/multivariate time series forecasting.

    For multivariate series (num_targets > 1), a separate Chronax
    SeasonalWindowAverage is fit for each target. The fitted instances are
    stored in ``self._models``.
    """

    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, ChronaxSeasonalWindowAverageHyperparams)
        self._models: list[SeasonalWindowAverage] = []

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
    ) -> SeasonalWindowAverage:
        """Fit a single Chronax SeasonalWindowAverage on one variate."""
        endog = jnp.asarray(y_context[:, 0], dtype=jnp.float32)
        model = SeasonalWindowAverage(
            season_length=int(self.season_length),
            window_size=int(self.window_size),
        )
        model.fit(endog)
        return model

    @validate_inputs
    def _predict(
        self,
        seasonal_window_average_model: SeasonalWindowAverage,
        y_context: np.ndarray,
        timestamps_context: np.ndarray,
        timestamps_target: np.ndarray,
        x_context: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> np.ndarray:
        """Predict future values for one variate with a fitted Chronax SeasonalWindowAverage."""
        if not self.is_fitted:
            raise ValueError(
                "ChronaxSeasonalWindowAverageModel not fitted. Call train() first."
            )

        if timestamps_target is None or len(timestamps_target) == 0:
            raise ValueError(
                "timestamps_target must be provided and non-empty for Chronax SeasonalWindowAverage prediction."
            )

        forecast_steps = int(len(timestamps_target))
        output = seasonal_window_average_model.predict(h=forecast_steps)
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
    ) -> "ChronaxSeasonalWindowAverageModel":
        """Train a separate Chronax SeasonalWindowAverage for each variate."""
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
            raise ValueError(
                "ChronaxSeasonalWindowAverageModel not fitted. Call train() first."
            )

        preds = []
        for idx, fitted_model in enumerate(self._models):
            prediction = self._predict(
                seasonal_window_average_model=fitted_model,
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


class SeasonalWindowAverageChronaxModel(ChronaxSeasonalWindowAverageModel):
    """Alias to satisfy TempusBench's ``{folder_name}_model.py`` class discovery."""

    pass
