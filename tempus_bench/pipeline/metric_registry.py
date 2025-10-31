"""
Model evaluation metric registry.

This module provides the MetricRegistry class which dynamically discovers and
registers evaluation metrics from the metrics directory. It computes metrics
based on model type (deterministic, stochastic, or hybrid) and handles both
point and probabilistic forecasting evaluation.
"""

import importlib
from typing import Any, Dict

import numpy as np

from ..metrics.base_metric import BaseMetric
from ..utils.log_manager import LogManager
from ..utils.paths import get_available_metrics


class MetricRegistry:
    """
    Registry for dynamically discovering and computing evaluation metrics.

    The MetricRegistry automatically discovers metric classes from the metrics
    directory and provides methods to compute all appropriate metrics based on
    the model type. It separates deterministic and stochastic metrics and
    applies them based on the model's capabilities.

    Attributes:
        metric_registry (Dict[str, BaseMetric]): Dictionary mapping metric names
            to metric instances.
        stochastic_metrics (List[str]): List of metric names that support stochastic
            predictions.
        deterministic_metrics (List[str]): List of metric names for point forecasts.
    """

    def __init__(self):
        """
        Initialize metric registry by discovering available metrics.

        This method scans the metrics directory, imports metric classes, and
        categorizes them as deterministic or stochastic based on their metric_type.
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
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics for model performance on given data.

        This method computes all appropriate metrics based on the model type. For
        deterministic models, only deterministic metrics are computed. For stochastic
        or hybrid models, both deterministic and stochastic metrics are computed.

        Args:
            y_true (np.ndarray): True target values with shape (num_steps, num_targets).
            y_pred (np.ndarray): Model predictions. Shape depends on model type:
                - Deterministic: (num_steps, num_targets)
                - Stochastic: (num_samples, num_steps, num_targets)
                - Hybrid: Tuple of (point_forecasts, samples)
            **kwargs (Dict[str, Any]): Additional keyword arguments for metrics:
                - model_type (str): 'deterministic', 'stochastic', or 'hybrid' (required)
                - point_forecast_statistic (str): Statistic for converting stochastic
                  to point forecasts (e.g., 'mean').
                - num_quantiles (int): Number of quantiles for quantile-based metrics.

        Returns:
            Dict[str, Any]: Dictionary mapping metric names to computed metric values.
                Values may be scalars, arrays, or nested dictionaries depending on the metric.

        Raises:
            ValueError: If model_type is not provided or is invalid.
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

        This method scans the metrics directory for Python files, imports metric
        classes that inherit from BaseMetric, and instantiates them. Metric names
        are derived from file names (e.g., mae.py -> Mae class -> 'mae' key).

        Returns:
            Dict[str, BaseMetric]: Dictionary mapping metric names (file stems)
                to instantiated metric instances. Only includes classes that
                inherit from BaseMetric but are not BaseMetric itself.
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

                # Try multiple naming conventions to find the metric class
                # Single-word metrics use all-uppercase (e.g., "mae" -> "MAE")
                # Multi-word metrics use CamelCase (e.g., "quantile_score" -> "QuantileScore")
                class_name_candidates = []

                # If single word, try uppercase first (e.g., "mae" -> "MAE")
                if "_" not in file_stem:
                    class_name_candidates.append(file_stem.upper())

                # Always try CamelCase (e.g., "mae" -> "Mae", "quantile_score" -> "QuantileScore")
                class_name_candidates.append(
                    "".join(word.capitalize() for word in file_stem.split("_"))
                )

                metric_class = None
                for class_name in class_name_candidates:
                    if hasattr(module, class_name):
                        metric_class = getattr(module, class_name)
                        break

                if metric_class is None:
                    # No matching class found in module
                    continue

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
