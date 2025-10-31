"""
Model evaluation.
"""

import importlib
from typing import Any, Dict

import numpy as np

from ..metrics.base_metric import BaseMetric
from ..utils.logger import LoggerManager
from ..utils.paths import get_available_metrics


class MetricRegistry:
    def __init__(self):
        """
        Initialize metric registry with configuration.

        Args:
            logger: Logger instance to use for logging (optional)
        """
        self.metric_registry = self._build_metric_registry()
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

    def _build_metric_registry(self) -> Dict[str, BaseMetric]:
        """
        Dynamically build metric registry from available metric files.

        Returns:
            Dict[str, BaseMetric]: Dictionary mapping metric names to metric instances
        """
        metric_registry = {}
        metric_files = get_available_metrics()

        for metric_file in metric_files:
            # Convert file path to module name
            # e.g., /path/to/tempus_bench/metrics/mae.py -> tempus_bench.metrics.mae
            file_stem = metric_file.stem  # e.g., "mae", "quantile_score"

            # Find the tempus_bench part in the path
            parts = metric_file.parts
            try:
                tempus_bench_idx = parts.index("tempus_bench")
                # Construct module name: tempus_bench.metrics.{file_stem}
                module_parts = parts[tempus_bench_idx:]
                module_name = ".".join(module_parts[:-1]) + "." + file_stem
            except ValueError:
                # Fallback: assume standard structure
                module_name = f"tempus_bench.metrics.{file_stem}"

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Convert file stem to CamelCase class name
                # e.g., "mae" -> "Mae", "quantile_score" -> "QuantileScore"
                class_name = "".join(word.capitalize() for word in file_stem.split("_"))

                # Get the class from the module
                metric_class = getattr(module, class_name)

                # Verify it's a subclass of BaseMetric
                if (
                    not issubclass(metric_class, BaseMetric)
                    or metric_class is BaseMetric
                ):
                    continue

                # Use file stem as the registry key (e.g., "mae", "quantile_score")
                # Each concrete metric class implements __init__ that handles initialization
                metric_registry[file_stem] = metric_class()  # type: ignore

            except (ImportError, AttributeError) as e:
                # Skip files that can't be imported or don't have valid metric classes
                continue

        return metric_registry
