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
from ..utils.logger import get_logger
class Evaluator:
    def __init__(self):
        """
        Initialize evaluator with configuration.

        Args:
            config: Configuration dictionary
            logs_path: Directory for storing log files (optional)
        """
        self.logger = get_logger()
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

        self.logger.debug("Evaluator", f"Evaluator initialized with Evaluation config: {self.eval_config}")
        self.logger.debug("Evaluator", f"Metrics to calculate: {self.metrics_to_calculate}")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Dict[str,Any]):
        """
        Evaluate model performance on given data.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Model predictions (point forecasts)
            **metric_kwargs: Additional keyword arguments for metrics.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if 'task_type' not in kwargs:
            raise ValueError("'task_type' must be provided in kwargs ('deterministic', 'stochastic')")

        task_type = kwargs['task_type']

        if task_type == 'deterministic':
            metrics_to_calculate = self.deterministic_metrics
        elif task_type == 'stochastic':
            metrics_to_calculate = self.stochastic_metrics + self.deterministic_metrics
        else:
            raise ValueError("'task_type' must be 'deterministic' or 'stochastic'")

        results = {}
        for metric in metrics_to_calculate:
            results[metric] = self.metric_registry[metric](
                y_true=y_true,
                y_pred=y_pred,
                task_type=task_type,
                point_forecast_statistic=kwargs['point_forecast_statistic'])
        return results
