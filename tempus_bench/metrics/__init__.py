"""
Metrics package for evaluating time series forecasting models.

This package provides various evaluation metrics for assessing the performance of
time series forecasting models, including both point forecasts and probabilistic forecasts.

Available Metrics:
- crps: Continuous Ranked Probability Score for stochastic forecasts
- interval_score: Interval score for prediction intervals (stochastic)
- quantile_loss: Quantile Loss (Pinball Loss) for stochastic forecasts
- mae: Mean Absolute Error (deterministic and stochastic)
- rmse: Root Mean Square Error (deterministic and stochastic)
- mape: Mean Absolute Percentage Error (deterministic only)
- mase: Mean Absolute Scaled Error (deterministic only)

Task Type Requirements:
- deterministic: mae, rmse, mape, mase
- stochastic: crps, quantile_loss, interval_score, mae, rmse

All metrics accept an optional 'task_type' parameter in kwargs to validate compatibility.
If not provided, metrics default to their compatible task type.

Usage:
    from benchmarking_pipeline.metrics import MAE, CRPS

    # Compute MAE (defaults to deterministic)
    mae = MAE()
    mae_score = mae(y_true, y_pred)

    # Compute MAE for stochastic forecasts (explicit)
    mae_score = mae(y_true, y_pred, task_type='stochastic')

    # Compute CRPS (defaults to stochastic)
    crps = CRPS()
    crps_score = crps(y_true, y_pred, y_pred_dist_samples=samples)
"""

__version__ = "1.0.0"
__description__ = "Evaluation metrics for time series forecasting models"
