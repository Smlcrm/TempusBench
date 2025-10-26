import pandas as pd
import numpy as np

from ..metrics.rmse import RMSE
from ..metrics.mae import MAE
from ..metrics.mase import MASE
from ..metrics.crps import CRPS
from ..metrics.quantile_loss import QuantileLoss
from ..metrics.interval_score import IntervalScore
from ..metrics.mape import MAPE
from ..utils.logger import get_logger

"""
Model evaluation.
"""

class Evaluator:
    def __init__(self, config=None):
        """
        Initialize evaluator with configuration.
        """
        self.config = config if config is not None else {}
        
        # Get logs_dir from config and initialize logger (fail fast, no defaults)
        logs_dir = self.config['logging']['logs_dir']
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
            "quantile_loss": QuantileLoss(),
            "interval_score": IntervalScore(),
        }

    def evaluate(self, y_predictions, y_true, y_train=None, **metric_kwargs):
        """
        Evaluate model performance on given data.

        Args:
            y_predictions (pd.Series or np.array): Model predictions.
            y_true (pd.Series or np.array): True target values.
            y_train (pd.Series or np.array, optional): Training target values for metrics like MASE.
            **metric_kwargs: Additional keyword arguments for metrics (e.g., y_pred_dist_samples for CRPS).

        Returns:
            Dictionary of evaluation metrics.
        """

        y_train_shape = y_train.shape if y_train is not None else "None"
        self.logger.debug("Evaluator",
            f"y_predictions shape: {y_predictions.shape}, y_true shape: {y_true.shape}, y_train shape: {y_train_shape}"
        )
        # If predictions longer than true, truncate to match
        if y_predictions is not None and y_true is not None:
            min_len = min(y_predictions.shape[0], y_true.shape[0])
            # Truncate along the time dimension (axis 0) for shape (num_steps, num_targets)
            y_predictions = y_predictions[:min_len]
            y_true = y_true[:min_len]

        results = {}
        for metric_name in self.metrics_to_calculate:
            if metric_name not in self.metric_registry:
                raise ValueError(
                    f"Metric '{metric_name}' is not recognized. Available metrics: {list(self.metric_registry.keys())}"
                )

            metric = self.metric_registry[metric_name]
            try:
                # Add task_type and freq to metric calls only if available in config
                metric_kwargs = {**metric_kwargs, 'task_type': self.task_type}
                if self.freq is not None:
                    metric_kwargs['freq'] = self.freq

                if metric_name == "mase":
                    if y_train is None:
                        raise ValueError(
                            "y_train must be provided for MASE calculation."
                        )
                    self.logger.debug("Evaluator",
                        f"Calculating MASE with y_true shape: {y_true.shape}, y_pred shape: {y_predictions.shape}, y_train shape: {y_train.shape}"
                    )
                    metric_value = metric(y_true, y_predictions, y_train=y_train, **metric_kwargs)
                    self.logger.debug("Evaluator", f"MASE result: {metric_value}")
                elif metric_name == "crps":
                    if "y_pred_dist_samples" not in metric_kwargs:
                        raise ValueError(
                            "y_pred_dist_samples must be provided for CRPS calculation."
                        )
                    metric_value = metric(y_true, y_predictions, **metric_kwargs)
                elif metric_name == "quantile_loss":
                    if (
                        "y_pred_quantiles" not in metric_kwargs
                        or "quantiles_q_values" not in metric_kwargs
                    ):
                        raise ValueError(
                            "y_pred_quantiles and quantiles_q_values must be provided for QuantileLoss calculation."
                        )
                    metric_value = metric(y_true, y_predictions, **metric_kwargs)
                elif metric_name == "interval_score":
                    if (
                        "y_pred_lower_bound" not in metric_kwargs
                        or "y_pred_upper_bound" not in metric_kwargs
                    ):
                        raise ValueError(
                            "y_pred_lower_bound and y_pred_upper_bound must be provided for IntervalScore calculation."
                        )
                    metric_value = metric(y_true, y_predictions, **metric_kwargs)
                else:
                    metric_value = metric(y_true, y_predictions, **metric_kwargs)

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
