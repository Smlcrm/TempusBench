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
from ..utils.logger import Logger


class Evaluator:
    def __init__(self, logger: Logger = None):
        """
        Initialize evaluator with configuration.

        Args:
            logger: Logger instance to use for logging (optional)
        """
        self.logger = logger
        self.metric_registry = {
            "rmse": RMSE(),
            "mae": MAE(),
            "mase": MASE(),
            "mape": MAPE(),
            "crps": CRPS(),
            "quantile_score": QuantileScore(),
            "weighted_interval_score": WeightedIntervalScore(),
        }
        self.stochastic_metrics = ["crps", "quantile_score", "weighted_interval_score"]
        self.deterministic_metrics = ["rmse", "mae", "mase", "mape"]

        if self.logger:
            self.logger.debug("Evaluator", "Evaluator initialized")

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Dict[str, Any]
    ):
        """
        Evaluate model performance on given data.

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
        for metric in metrics_to_calculate:
            results[metric] = self.metric_registry[metric](
                y_true=y_true,
                y_pred=y_pred,
                model_type=model_type,
                point_forecast_statistic=kwargs["point_forecast_statistic"],
                num_quantiles=kwargs["num_quantiles"],
            )
        return results
