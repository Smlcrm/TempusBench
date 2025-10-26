import pandas as pd
import numpy as np

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

        # Get evaluation metrics from the evaluation section of config (fail fast, no defaults)
        evaluation_cfg = self.config["evaluation"]
        self.metrics_to_calculate = evaluation_cfg["metrics"]

        # Get task type from config (fail fast, no defaults)
        self.task_type = self.config["task"]["task_type"]

        # Get frequency from dataset config (optional, may not exist)
        dataset_cfg = self.config["task"]["dataset"]
        self.freq = dataset_cfg.get("frequency")

        self.logger.debug("Evaluator", f"Evaluator initialized with config: {self.config}")
        self.logger.debug("Evaluator", f"Evaluation config: {evaluation_cfg}")
        self.logger.debug("Evaluator", f"Metrics to calculate: {self.metrics_to_calculate}")
        self.logger.debug("Evaluator", f"Task type: {self.task_type}")
        self.logger.debug("Evaluator", f"Frequency: {self.freq}")

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

        # Define which metrics are stochastic (require samples)
        self.stochastic_metrics = {"crps", "quantile_score", "weighted_interval_score"}

    def evaluate(self, y_predictions, y_true, y_train=None, y_pred_samples=None, **metric_kwargs):
        """
        Evaluate model performance on given data.

        Args:
            y_predictions (pd.Series or np.array): Model predictions (point forecasts).
            y_true (pd.Series or np.array): True target values.
            y_train (pd.Series or np.array, optional): Training target values for metrics like MASE.
            y_pred_samples (np.array, optional): Model prediction samples for stochastic metrics.
            **metric_kwargs: Additional keyword arguments for metrics.

        Returns:
            Dictionary of evaluation metrics.
        """

        y_train_shape = y_train.shape if y_train is not None else "None"
        y_pred_samples_shape = y_pred_samples.shape if y_pred_samples is not None else "None"
        self.logger.debug("Evaluator",
            f"y_predictions shape: {y_predictions.shape}, y_true shape: {y_true.shape}, y_train shape: {y_train_shape}, y_pred_samples shape: {y_pred_samples_shape}"
        )
        # If predictions longer than true, truncate to match
        if y_predictions is not None and y_true is not None:
            min_len = min(y_predictions.shape[0], y_true.shape[0])
            # Truncate along the time dimension (axis 0) for shape (num_steps, num_targets)
            y_predictions = y_predictions[:min_len]
            y_true = y_true[:min_len]

        # If samples provided, truncate them too
        if y_pred_samples is not None and y_true is not None:
            min_len = min(y_pred_samples.shape[1], y_true.shape[0])
            # Truncate along the time dimension (axis 1) for shape (num_samples, num_steps, num_targets)
            y_pred_samples = y_pred_samples[:, :min_len]

        results = {}
        for metric_name in self.metrics_to_calculate:
            if metric_name not in self.metric_registry:
                raise ValueError(
                    f"Metric '{metric_name}' is not recognized. Available metrics: {list(self.metric_registry.keys())}"
                )

            metric = self.metric_registry[metric_name]
            try:
                # Add task_type and freq to metric calls only if available in config
                metric_kwargs_copy = {**metric_kwargs, 'task_type': self.task_type}
                if self.freq is not None:
                    metric_kwargs_copy['freq'] = self.freq

                # Handle stochastic metrics
                if metric_name in self.stochastic_metrics:
                    if y_pred_samples is None:
                        raise ValueError(
                            f"y_pred_samples must be provided for stochastic metric '{metric_name}' calculation."
                        )
                    metric_kwargs_copy['y_pred_samples'] = y_pred_samples
                    metric_value = metric(y_true, y_predictions, **metric_kwargs_copy)
                
                # Handle deterministic metrics
                else:
                    if metric_name == "mase":
                        if y_train is None:
                            raise ValueError(
                                "y_train must be provided for MASE calculation."
                            )
                        self.logger.debug("Evaluator",
                            f"Calculating MASE with y_true shape: {y_true.shape}, y_pred shape: {y_predictions.shape}, y_train shape: {y_train.shape}"
                        )
                        metric_value = metric(y_true, y_predictions, y_train=y_train, **metric_kwargs_copy)
                        self.logger.debug("Evaluator", f"MASE result: {metric_value}")
                    else:
                        metric_value = metric(y_true, y_predictions, **metric_kwargs_copy)

                # If the metric returns a dict, merge it into results
                if isinstance(metric_value, dict):
                    results.update(metric_value)
                else:
                    results[metric_name] = metric_value

            except Exception as e:
                self.logger.error("Evaluator", f"Failed to calculate {metric_name}: {e}")
                # Continue with other metrics instead of failing completely
                continue

        return results
