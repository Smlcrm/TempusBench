import numpy as np

from ..utils.utils import compute_point_forecast


class BaseMetric:

    def __init__(self, metric_type: str):
        self.metric_type = metric_type

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        """
        Process y_pred based on model type and delegate to _compute method.

        This method ensures all subclasses automatically get preprocessed y_pred
        without needing to manually call process_y_pred.
        """
        # Extract required kwargs for process_y_pred
        model_type = kwargs.get("model_type")
        if model_type is None:
            raise ValueError("'model_type' must be provided in kwargs")

        point_forecast_statistic = kwargs.get("point_forecast_statistic")

        # Process y_pred before passing to subclass implementation
        processed_y_pred = self.process_y_pred(
            y_true=y_true,
            y_pred=y_pred,
            model_type=model_type,
            point_forecast_statistic=point_forecast_statistic,
        )

        # Delegate to subclass implementation
        return self._compute(y_true=y_true, y_pred=processed_y_pred, **kwargs)

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        """
        Subclass-specific metric computation.

        Subclasses should implement this method instead of __call__.
        This method receives preprocessed y_pred that has been processed
        by process_y_pred based on the model type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _compute method"
        )

    def process_y_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: str,
        point_forecast_statistic: str | None = None,
    ) -> np.ndarray:
        """Process y_pred based on model type, applying necessary transformations."""
        self._validate_model_type(model_type)

        if model_type == "deterministic":
            length, num_y_features = self._validate_deterministic_inputs(y_true, y_pred)
            forecast = y_pred

        elif model_type == "stochastic":
            num_samples, length, num_y_features = self._validate_stochastic_inputs(
                y_true, y_pred
            )
            if self.metric_type == "deterministic":
                self._validate_point_forecast_statistic(point_forecast_statistic)
                # After validation, point_forecast_statistic is guaranteed to be str
                assert point_forecast_statistic is not None
                forecast = compute_point_forecast(y_pred, point_forecast_statistic)
            else:
                forecast = y_pred

        elif model_type == "hybrid":
            length, num_y_features = self._validate_hybrid_inputs(y_true, y_pred)
            point_forecasts, forecast_samples = y_pred
            if self.metric_type == "deterministic":
                forecast = point_forecasts
            else:
                forecast = forecast_samples

        return forecast

    ###################################################### Validation methods #########################################################

    @staticmethod
    def _validate_model_type(model_type: str) -> None:
        """Validate that model_type is one of the supported values."""
        valid_types = {"deterministic", "stochastic", "hybrid"}
        if model_type not in valid_types:
            raise ValueError(
                f"Invalid model type: {model_type}. " f"Must be one of {valid_types}"
            )

    @staticmethod
    def _validate_deterministic_inputs(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[int, int]:
        """
        Validate inputs for deterministic model type.

        Args:
            y_true: True values array
            y_pred: Predictions array

        Returns:
            tuple[int, int]: (length, num_y_features) extracted from y_pred shape

        Raises:
            ValueError: If shapes are invalid or don't match
        """
        # Validate y_pred is 2D: (num_steps, num_targets)
        if len(y_pred.shape) != 2:
            raise ValueError(
                f"Expected 2D y_pred for deterministic model, got shape {y_pred.shape}"
            )
        length, num_y_features = y_pred.shape

        # Validate y_true shape matches y_pred
        if y_true.shape != (length, num_y_features):
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, but expected "
                f"({length}, {num_y_features}) to match y_pred"
            )

        return length, num_y_features

    @staticmethod
    def _validate_stochastic_inputs(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[int, int, int]:
        """
        Validate inputs for stochastic model type.

        Args:
            y_true: True values array
            y_pred: Predictions array

        Returns:
            tuple[int, int, int]: (num_samples, length, num_y_features) extracted from y_pred shape

        Raises:
            ValueError: If shapes are invalid or don't match
        """
        # Validate y_pred is 3D: (num_samples, num_steps, num_targets)
        if len(y_pred.shape) != 3:
            raise ValueError(
                f"Expected 3D y_pred for stochastic model, got shape {y_pred.shape}"
            )
        num_samples, length, num_y_features = y_pred.shape

        # Validate y_true shape matches y_pred
        if y_true.shape != (length, num_y_features):
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, but expected "
                f"({length}, {num_y_features}) to match y_pred "
                f"(num_samples={num_samples}, time_steps={length}, num_targets={num_y_features})"
            )

        return num_samples, length, num_y_features

    @staticmethod
    def _validate_hybrid_inputs(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[int, int]:
        """
        Validate inputs for hybrid model type.

        Args:
            y_true: True values array
            y_pred: Tuple of (point_forecasts, forecast_samples)

        Returns:
            tuple[int, int]: (length, num_y_features) extracted from point_forecasts shape

        Raises:
            ValueError: If types/shapes are invalid or don't match
        """
        # Validate y_pred is a tuple with 2 elements
        if not isinstance(y_pred, tuple) or len(y_pred) != 2:
            raise ValueError(
                f"Expected tuple of (point_forecasts, forecast_samples) for hybrid model, "
                f"got {type(y_pred)}"
            )
        point_forecasts, forecast_samples = y_pred

        # Validate point_forecasts is 2D
        if len(point_forecasts.shape) != 2:
            raise ValueError(
                f"Expected 2D point_forecasts for hybrid model, got shape {point_forecasts.shape}"
            )

        # Validate forecast_samples is 3D
        if len(forecast_samples.shape) != 3:
            raise ValueError(
                f"Expected 3D forecast_samples for hybrid model, got shape {forecast_samples.shape}"
            )

        length, num_y_features = point_forecasts.shape
        num_samples, length_samples, num_y_features_samples = forecast_samples.shape

        # Validate dimensions match between point_forecasts and forecast_samples
        if length != length_samples or num_y_features != num_y_features_samples:
            raise ValueError(
                f"Shape mismatch in hybrid model: point_forecasts has shape {point_forecasts.shape}, "
                f"but forecast_samples has shape {forecast_samples.shape}"
            )

        # Validate y_true shape matches
        if y_true.shape != (length, num_y_features):
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, but expected "
                f"({length}, {num_y_features}) to match hybrid predictions"
            )

        return length, num_y_features

    @staticmethod
    def _validate_point_forecast_statistic(
        point_forecast_statistic: str | None,
    ) -> None:
        """Validate that point_forecast_statistic is provided when required."""
        if point_forecast_statistic is None:
            raise ValueError(
                "point_forecast_statistic must be provided when using "
                "deterministic metrics with stochastic models"
            )
