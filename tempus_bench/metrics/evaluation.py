import pandas as pd
import numpy as np
from typing import Dict, Any

from ..metrics.rmse import RMSE
from ..metrics.mae import MAE
from ..metrics.mase import MASE
from ..metrics.crps import CRPS
from ..metrics.quantile_score import QuantileScore
from ..metrics.weighted_interval_score import WeightedIntervalScore
from ..metrics.mape import MAPE
from ..utils.logger import get_logger

"""
Model evaluation.
"""
class Evaluator:
    def __init__(self, config=None, logs_dir: str = None):
        """
        Initialize evaluator with configuration.

        Args:
            config: Configuration dictionary
            logs_dir: Directory for storing log files (optional)
        """
        self.config = config if config is not None else {}

        # Get logs_dir from parameter, fail fast if not provided
        if logs_dir is None:
            raise ValueError("logs_dir parameter is required")
        self.logger = get_logger(logs_dir)
        evaluation_cfg = self.config["evaluation"]
        self.metrics_to_calculate = evaluation_cfg["metrics"]
        self.task_type = self.config["task"]["task_type"]

        self.logger.debug("Evaluator", f"Evaluator initialized with config: {self.config}")
        self.logger.debug("Evaluator", f"Evaluation config: {evaluation_cfg}")
        self.logger.debug("Evaluator", f"Metrics to calculate: {self.metrics_to_calculate}")
        self.logger.debug("Evaluator", f"Task type: {self.task_type}")

        # maps string names to metric class instances
        self.metric_registry = {
            "rmse": RMSE(),
            "mae": MAE(),
            "mase": MASE(),
            "mape": MAPE(),
            "crps": CRPS(),
            "quantile_score": QuantileScore(),
            "weighted_interval_score": WeightedIntervalScore(),
        }

        self.stochastic_metrics = {"crps", "quantile_score", "weighted_interval_score"}
        self.deterministic_metrics = {"rmse", "mae", "mase", "mape"}

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
        results = {}
        for metric in self.metrics_to_calculate:
            if metric in self.stochastic_metrics:
                kwargs = {
                    'task_type' : 'stochastic',
                    'point_forecast_statistic' : self.config['evaluation']['point_forecast_statistic']
                }
            elif metric in self.deterministic_metrics:
                kwargs = {'task_type' : 'deterministic'}
            else:
                raise ValueError(f"Metric '{metric}' is not a recognized deterministic or stochastic metric.")
            results[metric] = self.metric_registry[metric](y_true, y_pred, **kwargs)

        return results
