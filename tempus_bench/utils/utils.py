import numpy as np

def compute_point_forecast(y_pred: np.ndarray, point_forecast_statistic: str) -> np.ndarray:
    """
    Compute the point forecast from the predicted samples.

    Args:
        y_pred: Prediction array with shape (num_samples, time_steps, num_targets) for stochastic
            or (time_steps, num_targets) for deterministic.
        point_forecast_statistic: The statistic to apply. Supported values:
            - "mean": Compute mean across the first dimension (samples)
            - "median": Compute median across the first dimension (samples)

    Returns:
        Point forecast array with shape (time_steps, num_targets)

    Raises:
        ValueError: If point_forecast_statistic is not supported
    """
    # Handle deterministic predictions (already point forecasts)
    if y_pred.ndim == 2:
        return y_pred

    # Handle stochastic predictions (num_samples, time_steps, num_targets)
    if y_pred.ndim != 3:
        raise ValueError(
            f"Expected y_pred to be 2D (deterministic) or 3D (stochastic), "
            f"got {y_pred.ndim}D with shape {y_pred.shape}"
        )

    if point_forecast_statistic == "mean":
        return np.mean(y_pred, axis=0)
    elif point_forecast_statistic == "median":
        return np.median(y_pred, axis=0)
    else:
        raise ValueError(
            f"Invalid point forecast statistic: {point_forecast_statistic}"
        )
