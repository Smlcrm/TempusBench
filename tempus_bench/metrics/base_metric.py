"""
Base metric class for evaluation metrics.

This module provides the BaseMetric abstract base class that all evaluation
metrics must inherit from. It handles processing of different prediction types
(deterministic, stochastic, hybrid) and provides validation methods.
"""

import numpy as np


class BaseMetric:
    """
    Base class for all evaluation metrics.

    This abstract base class defines the interface for evaluation metrics and
    provides utilities for processing predictions from different model types
    (deterministic, stochastic, hybrid).

    Attributes:
        metric_type (str): Type of metric, either "deterministic" or "stochastic".
            Determines which model types this metric can evaluate.
    """

    def __init__(self, metric_type: str):
        """
        Initialize base metric with metric type.

        Args:
            metric_type (str): Type of metric, either "deterministic" for point
                forecast metrics or "stochastic" for probabilistic metrics.
        """
        self.metric_type = metric_type

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        """
        Compute the metric value.

        Subclasses must implement this method to compute the actual metric value.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Model predictions (processed by process_y_pred).
            **kwargs: Additional arguments for metric computation.

        Returns:
            Any: Computed metric value (scalar, array, or dictionary).

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        pass

    def process_y_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        point_forecast_statistic: str | None = None,
    ) -> np.ndarray:
        """
        Process model predictions based on model type and metric type.

        This method handles conversion of stochastic or hybrid predictions to
        the format required by the metric type (deterministic or stochastic).

        Args:
            y_true (np.ndarray): True target values with shape (length, num_y_features).
            y_pred (np.ndarray): Model predictions. Shape depends on model type:
                - Deterministic: (length, num_y_features)
                - Stochastic: (num_samples, length, num_y_features)
                - Hybrid: Tuple of (point_forecasts, samples)
            model_type (str): Type of model, one of "deterministic", "stochastic", "hybrid".
            point_forecast_statistic (Optional[str]): Statistic to use for converting
                stochastic predictions to point forecasts (e.g., "mean"). Required
                if model_type is "stochastic" and metric_type is "deterministic".

        Returns:
            np.ndarray: Processed predictions in the format required by the metric.
                Shape is (length, num_y_features) for deterministic metrics or
                (num_samples, length, num_y_features) for stochastic metrics.

        Raises:
            ValueError: If model_type is invalid, shapes don't match, or required
                parameters are missing.
        """
        num_samples, length, num_y_features = y_pred.shape

        # Validate shapes
        self._validate_y_true_shape(y_true, length, num_y_features, num_samples)
        self._validate_y_pred_dimensions(y_pred)

        if model_type == "deterministic":
            # do nothing
            forecast = y_pred

        elif model_type == "stochastic":
            if self.metric_type == "deterministic":
                # process stochastic predictions by applying the specified statistic
                forecast = self._process_stochastic_prediction(
                    y_pred, point_forecast_statistic
                )
            else:
                # do nothing
                forecast = y_pred

        elif model_type == "hybrid":
            point_forecasts, forecast_samples = y_pred
            if self.metric_type == "deterministic":
                forecast = point_forecasts
            else:
                forecast = forecast_samples

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        return forecast

    ###################################################### Helper functions #########################################################

    @staticmethod
    def _validate_y_true_shape(
        y_true: np.ndarray, length: int, num_y_features: int, num_samples: int
    ) -> None:
        """
        Validate that y_true shape matches expected dimensions from y_pred.

        Args:
            y_true (np.ndarray): True target values to validate.
            length (int): Expected number of time steps.
            num_y_features (int): Expected number of target features.
            num_samples (int): Number of samples (for stochastic predictions).

        Raises:
            ValueError: If y_true shape doesn't match expected dimensions.
        """
        if y_true.shape != (length, num_y_features):
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, but expected "
                f"({length}, {num_y_features}) to match y_pred "
                f"(num_samples={num_samples}, time_steps={length}, num_targets={num_y_features})"
            )

    @staticmethod
    def _validate_y_pred_dimensions(y_pred: np.ndarray) -> None:
        """
        Validate that y_pred doesn't have more than 2 dimensions for deterministic evaluation.

        Args:
            y_pred (np.ndarray): Prediction array to validate.

        Raises:
            ValueError: If y_pred has more than 2 dimensions for deterministic evaluation.
        """
        if len(y_pred.shape) > 2:
            raise ValueError(
                "y_pred can't have more than 2 dimensions for deterministic evaluation"
            )

    @staticmethod
    def _process_stochastic_prediction(
        y_pred: np.ndarray, point_forecast_statistic: str | None = None
    ) -> np.ndarray:
        """
        Process stochastic predictions by applying the specified statistic.

        This method converts stochastic predictions (samples) to point forecasts
        by applying a statistic like mean across the samples dimension.

        Args:
            y_pred (np.ndarray): Prediction array with shape (num_samples, time_steps, num_targets).
            point_forecast_statistic (Optional[str]): The statistic to apply.
                Currently only 'mean' is supported.

        Returns:
            np.ndarray: Processed prediction array with shape (time_steps, num_targets),
                representing point forecasts.

        Raises:
            ValueError: If point_forecast_statistic is None or unsupported.
        """
        if point_forecast_statistic is None:
            raise ValueError(
                "point_forecast_statistic must be provided for stochastic evaluation"
            )

        if point_forecast_statistic == "mean":
            return np.mean(y_pred, axis=0)
        else:
            raise ValueError(
                "RMSE can only handle point_forecast_statistic == 'mean' for stochastic evaluation."
            )
