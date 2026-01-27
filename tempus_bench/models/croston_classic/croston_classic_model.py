"""
Croston's Classic Model implementation for intermittent demand forecasting.
"""

from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from tempus_bench.models.base_model import BaseModel, validate_inputs


class CrostonClassicHyperparams(PydanticBaseModel):
    # Highly Influential Hyperparameters
    alpha: float = Field(
        ..., gt=0, lt=1, description="Smoothing parameter for demand level"
    )
    gamma: float = Field(
        ..., gt=0, lt=1, description="Smoothing parameter for interval level"
    )


class CrostonClassicModel(BaseModel):
    def __init__(self, params: Dict[str, Any], settings: Dict[str, Any]):
        super().__init__(params, settings, CrostonClassicHyperparams)  # type: ignore

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
    ) -> "CrostonClassicModel":
        """
        Train the Croston's Classic model on the given time series data.

        Decomposes the series into non-zero demand values and the intervals between them,
        and applies Simple Exponential Smoothing (SES) to both.

        Args:
            y_context (np.ndarray): Past target values for training (shape: [n_timesteps, n_variates]).
            y_target (np.ndarray): Ignored; included for interface compatibility.
            timestamps_context (np.ndarray): Ignored.
            timestamps_target (np.ndarray): Ignored.
            **kwargs: Expected to include 'alpha' and 'gamma'.

        Returns:
            CrostonClassicModel: Trained model.
        """
        alpha = self.alpha  # type: ignore
        gamma = self.gamma  # type: ignore
        num_targets = y_context.shape[1]

        demand_levels = np.zeros(num_targets)
        interval_levels = np.zeros(num_targets)

        for target in range(num_targets):
            serie = y_context[:, target]
            non_zero_idx = np.nonzero(serie)[0]

            if len(non_zero_idx) < 2:
                demand_levels[target] = np.mean(serie) if serie.size > 0 else 0.0
                interval_levels[target] = len(serie) if serie.size > 0 else 1.0
                continue

            demands = serie[non_zero_idx]
            intervals = np.diff(non_zero_idx)

            # Initialize with the first non-zero demand and interval
            current_demand_level = demands[0]
            current_interval_level = intervals[0] if intervals.size > 0 else 1

            # Simple Exponential Smoothing for demands
            for d in demands[1:]:
                current_demand_level = alpha * d + (1 - alpha) * current_demand_level

            # Simple Exponential Smoothing for intervals
            for v in intervals[1:]:
                current_interval_level = (
                    gamma * v + (1 - gamma) * current_interval_level
                )

            demand_levels[target] = current_demand_level
            interval_levels[target] = current_interval_level

        self._demand_level = demand_levels
        self._interval_level = interval_levels
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
        """
        Predict future values using the trained Croston's Classic model.

        Args:
            y_context (np.ndarray): Ignored.
            timestamps_context (np.ndarray): Ignored.
            timestamps_target (np.ndarray): Used to determine forecast horizon.
            **kwargs: Ignored.

        Returns:
            np.ndarray: Predictions with shape (forecast_horizon, num_targets).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        forecast_horizon = len(timestamps_target)
        forecast = np.divide(
            self._demand_level,
            self._interval_level,
            out=np.zeros_like(self._demand_level, dtype=float),
            where=self._interval_level != 0,
        )
        forecast = np.tile(forecast, (forecast_horizon, 1))
        return forecast

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the Croston's Classic model.
        """
        summary = {
            "model_type": "CrostonClassic",
            "alpha": self.alpha,  # type: ignore
            "gamma": self.gamma,  # type: ignore
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            summary["fitted_demand_level"] = getattr(self, "_demand_level", None)
            summary["fitted_interval_level"] = getattr(self, "_interval_level", None)

        return summary
