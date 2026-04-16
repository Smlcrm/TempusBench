import numpy as np


def validate_forecast_sanity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    point_forecast_statistic: str = "mean",
) -> dict:
    """
    Run sanity checks on forecasts to detect obvious issues.

    Returns a dict with keys: ok, warnings, stats.
    """
    result = {"ok": True, "warnings": [], "stats": {}}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. Check for NaN/inf
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        result["ok"] = False
        result["warnings"].append("Predictions contain NaN or Inf")
        return result

    # 2. Compute point forecast for comparison
    point_pred = compute_point_forecast(y_pred, point_forecast_statistic)
    result["stats"]["point_pred_mean"] = float(np.mean(point_pred))
    result["stats"]["point_pred_std"] = float(np.std(point_pred))
    result["stats"]["y_true_mean"] = float(np.mean(y_true))
    result["stats"]["y_true_std"] = float(np.std(y_true))

    # 3. Scale check: prediction std should be in same order as y_true (both normalized)
    if result["stats"]["y_true_std"] > 1e-8:
        scale_ratio = result["stats"]["point_pred_std"] / result["stats"]["y_true_std"]
        if scale_ratio < 0.01:
            result["warnings"].append(
                f"Predictions have much smaller variance than y_true (ratio={scale_ratio:.4f})"
            )
        elif scale_ratio > 100:
            result["warnings"].append(
                f"Predictions have much larger variance than y_true (ratio={scale_ratio:.2f})"
            )

    # 4. Correlation check (flatten for single correlation)
    y_true_flat = y_true.flatten()
    point_flat = point_pred.flatten()
    if len(y_true_flat) > 1 and np.std(y_true_flat) > 1e-8 and np.std(point_flat) > 1e-8:
        corr = np.corrcoef(y_true_flat, point_flat)[0, 1]
        result["stats"]["correlation"] = float(corr)
        if np.isnan(corr) or (corr < -0.9):
            result["warnings"].append(
                f"Point forecast negatively correlated with y_true (corr={corr:.3f})"
            )

    # 5. Model MAE (for logging)
    if y_true.size > 0:
        mae_model = float(np.mean(np.abs(y_true - point_pred)))
        result["stats"]["mae_model"] = mae_model

    if result["warnings"]:
        result["ok"] = False
    return result


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
