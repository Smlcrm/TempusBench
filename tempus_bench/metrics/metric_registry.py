"""
Model evaluation.
"""

from typing import Any, Dict

import numpy as np

from ..metrics.crps import CRPS
from ..metrics.mae import MAE
from ..metrics.mape import MAPE
from ..metrics.mase import MASE
from ..metrics.quantile_score import QuantileScore
from ..metrics.rmse import RMSE
from ..metrics.weighted_interval_score import WeightedIntervalScore
from ..utils.logger import LoggerManager


class MetricRegistry:
    def __init__(self):
        """
        Initialize metric registry with configuration.

        Args:
            logger: Logger instance to use for logging (optional)
        """
        self.metric_registry = {
            "rmse": RMSE(),
            "mae": MAE(),
            "mase": MASE(),
            "mape": MAPE(),
            "crps": CRPS(),
            "quantile_score": QuantileScore(),
            "weighted_interval_score": WeightedIntervalScore(),
        }
        self.stochastic_metrics = [
            metric_name
            for metric_name, metric in self.metric_registry.items()
            if metric.metric_type == "stochastic"
        ]
        self.deterministic_metrics = [
            metric_name
            for metric_name, metric in self.metric_registry.items()
            if metric.metric_type == "deterministic"
        ]

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Dict[str, Any]
    ):
        """
        Compute evaluation metrics for model performance on given data.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Model predictions (point forecasts)
            **metric_kwargs: Additional keyword arguments for metrics.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if "model_type" not in kwargs:
            raise ValueError(
                "'model_type' must be provided in kwargs ('deterministic', 'stochastic', 'hybrid')"
            )

        model_type = kwargs["model_type"]

        if model_type == "deterministic":
            metrics_to_calculate = self.deterministic_metrics
        elif model_type in ("stochastic", "hybrid"):
            metrics_to_calculate = self.stochastic_metrics + self.deterministic_metrics
        else:
            raise ValueError(
                "'model_type' must be 'deterministic', 'stochastic', or 'hybrid'"
            )

        results = {}
        for metric_name in metrics_to_calculate:
            results[metric_name] = self.metric_registry[metric_name](
                y_true=y_true,
                y_pred=y_pred,
                model_type=model_type,
                point_forecast_statistic=kwargs["point_forecast_statistic"],
                num_quantiles=kwargs["num_quantiles"],
            )
        return results
