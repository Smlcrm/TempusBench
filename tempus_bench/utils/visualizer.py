"""
Visualization utilities for the benchmarking pipeline.

The Visualizer class supports Matplotlib figures where needed, and TensorBoard
scalar series for forecast vs actual (hierarchical tags: model / task / window).
"""

from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from ..utils.log_manager import LogManager


class Visualizer:
    """
    Creates visualizations for time series forecasting results.

    The Visualizer provides methods for prediction plots (Matplotlib) and
    logging forecast comparisons to TensorBoard as scalar series (no PNGs).
    """

    def __init__(self):
        plt.style.use("fivethirtyeight")
        sns.set_theme(style="whitegrid")

    def plot_predictions(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        time_index: Optional[Union[pd.DatetimeIndex, list]] = None,
        y_pred_lower: Optional[Union[pd.Series, np.ndarray]] = None,
        y_pred_upper: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "Predictions vs Actual Values",
        save_path: Optional[str] = None,
    ):
        plt.figure(figsize=(12, 6))

        if isinstance(y_true, pd.Series):
            time_index = time_index if time_index is not None else y_true.index
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        x = time_index if time_index is not None else np.arange(len(y_true))

        plt.plot(x, y_true, label="Actual", color="blue", alpha=0.7)
        plt.plot(x, y_pred, label="Predicted", color="red", alpha=0.7)

        if y_pred_lower is not None and y_pred_upper is not None:
            if isinstance(y_pred_lower, pd.Series):
                y_pred_lower = y_pred_lower.values
            if isinstance(y_pred_upper, pd.Series):
                y_pred_upper = y_pred_upper.values

            plt.fill_between(
                x,
                y_pred_lower,
                y_pred_upper,
                color="red",
                alpha=0.2,
                label="Prediction Interval",
            )

        plt.title(title)
        plt.xlabel("Time" if isinstance(x, pd.DatetimeIndex) else "Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if isinstance(x, pd.DatetimeIndex):
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            LogManager.get_logger().info("Visualizer", f"Plot saved to {save_path}")

        plt.show()

    def plot_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        title: str = "Residual Analysis",
        save_path: Optional[str] = None,
    ):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color="blue")
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted Values")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color="green", edgecolor="black")
        axes[0, 1].set_xlabel("Residual Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Residuals")
        axes[0, 1].grid(True, alpha=0.3)

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normality Check)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(residuals, alpha=0.7, color="purple")
        axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("Time Index")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals Over Time")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, y=1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            LogManager.get_logger().info("Visualizer", f"Plot saved to {save_path}")

        plt.show()

    def plot_forecast_window(
        self,
        *,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        timestamps_pred: np.ndarray,
        model_name: str,
        task_name: str,
        hyperparameters: Optional[Mapping[str, object]] = None,
        window_idx: int = 0,
    ) -> None:
        """
        Log actual vs predicted as TensorBoard scalars (no image files).

        See :meth:`LogManager.log_forecast_window_scalars`. Tags group by **model**,
        then **task**, **forecast origin** (first ``timestamps_pred``), and a stable
        **hyperparameter trial** segment when tuning so multiple grid points do not
        collide in TensorBoard.

        Args:
            hyperparameters: Params used to produce ``y_true``/``y_pred`` (omit or
                ``{}`` for a single-configuration model).
        """
        hp = {} if hyperparameters is None else dict(hyperparameters)
        try:
            if y_true.ndim != 2 or y_pred.ndim != 2:
                raise ValueError(
                    f"y_true and y_pred must be 2D arrays, got {y_true.shape} and {y_pred.shape}"
                )
            if y_true.shape != y_pred.shape:
                raise ValueError(
                    f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
                )
            if timestamps_pred.ndim != 1 or timestamps_pred.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    f"Invalid shape for timestamps_pred: expected 1D array with length {y_pred.shape[0]}, "
                    f"but got shape {timestamps_pred.shape}."
                )

            t0 = np.asarray(timestamps_pred).reshape(-1)[0]
            LogManager.get_logger().log_forecast_window_scalars(
                task_name=task_name,
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                forecast_start_timestamp=t0,
                hyperparameters=hp,
            )
        except Exception as e:
            LogManager.get_logger().error(
                "Visualizer",
                f"Error logging forecast scalars for {model_name} task={task_name}: {e}",
            )
            raise
