import numpy as np


class BaseMetric:

    def __init__(self, metric_type: str):
        self.metric_type = metric_type

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        pass

    def process_y_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        point_forecast_statistic: str | None = None,
    ) -> np.ndarray:
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
        """Validate that y_true shape matches expected dimensions from y_pred."""
        if y_true.shape != (length, num_y_features):
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, but expected "
                f"({length}, {num_y_features}) to match y_pred "
                f"(num_samples={num_samples}, time_steps={length}, num_targets={num_y_features})"
            )

    @staticmethod
    def _validate_y_pred_dimensions(y_pred: np.ndarray) -> None:
        """Validate that y_pred doesn't have more than 2 dimensions for deterministic evaluation."""
        if len(y_pred.shape) > 2:
            raise ValueError(
                "y_pred can't have more than 2 dimensions for deterministic evaluation"
            )

    @staticmethod
    def _process_stochastic_prediction(
        y_pred: np.ndarray, point_forecast_statistic: str | None = None
    ) -> np.ndarray:
        """Process stochastic predictions by applying the specified statistic.

        Args:
            y_pred: Prediction array with shape (num_samples, time_steps, num_targets)
            point_forecast_statistic: The statistic to apply ('mean' is currently supported)

        Returns:
            Processed prediction array

        Raises:
            ValueError: If point_forecast_statistic is None or unsupported
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
