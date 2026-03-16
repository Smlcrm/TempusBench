"""
Visualization utilities for the benchmarking pipeline.

This module provides the Visualizer class for creating plots and visualizations
of time series forecasts, including prediction plots with confidence intervals
and residual analysis plots.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from ..utils.log_manager import LogManager


class Visualizer:
    """
    Creates visualizations for time series forecasting results.

    The Visualizer provides methods to create various plots including prediction
    vs actual values, confidence intervals, and residual analysis. It supports
    both deterministic and probabilistic forecasts.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with visualization parameters.
        logger (Optional[LoggerManager]): Logger instance for logging operations.
    """

    def __init__(self):
        """
        Initialize visualizer with configuration.

        """
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
        """
        Plot predictions against actual values with optional confidence intervals.

        This method creates a line plot showing actual values, predicted values,
        and optionally prediction intervals. The plot supports both pandas and
        numpy array inputs.

        Args:
            y_true (Union[pd.Series, np.ndarray]): Actual target values.
            y_pred (Union[pd.Series, np.ndarray]): Predicted values.
            time_index (Optional[Union[pd.DatetimeIndex, list]]): Optional time index
                for x-axis. If None and y_true is a pd.Series, uses its index.
            y_pred_lower (Optional[Union[pd.Series, np.ndarray]]): Lower bound of
                prediction interval for probabilistic forecasts.
            y_pred_upper (Optional[Union[pd.Series, np.ndarray]]): Upper bound of
                prediction interval for probabilistic forecasts.
            title (str): Plot title. Defaults to "Predictions vs Actual Values".
            save_path (Optional[str]): If provided, saves the plot to this path.

        Returns:
            None: Displays the plot using matplotlib.pyplot.show().
        """
        plt.figure(figsize=(12, 6))

        # Convert inputs to numpy arrays if they're pandas objects
        if isinstance(y_true, pd.Series):
            time_index = time_index if time_index is not None else y_true.index
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Create x-axis values
        x = time_index if time_index is not None else np.arange(len(y_true))

        # Plot actual values
        plt.plot(x, y_true, label="Actual", color="blue", alpha=0.7)

        # Plot predictions
        plt.plot(x, y_pred, label="Predicted", color="red", alpha=0.7)

        # Plot confidence intervals if available
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

        # Customize plot
        plt.title(title)
        plt.xlabel("Time" if isinstance(x, pd.DatetimeIndex) else "Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rotate x-axis labels if using datetime
        if isinstance(x, pd.DatetimeIndex):
            plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot if path provided
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
        """
        Plot residual analysis including residual distribution and Q-Q plot.

        This method creates a two-panel plot showing the distribution of residuals
        and a Q-Q plot to assess normality of residuals.

        Args:
            y_true (Union[pd.Series, np.ndarray]): Actual target values.
            y_pred (Union[pd.Series, np.ndarray]): Predicted values.
            title (str): Plot title. Defaults to "Residual Analysis".
            save_path (Optional[str]): If provided, saves the plot to this path.

        Returns:
            None: Displays the plot using matplotlib.pyplot.show().
        """
        # Convert inputs to numpy arrays if they're pandas objects
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        residuals = y_true - y_pred

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot residual distribution
        sns.histplot(residuals, kde=True, ax=ax1)
        ax1.set_title("Residual Distribution")
        ax1.set_xlabel("Residual Value")
        ax1.set_ylabel("Count")

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot")

        # Set main title
        fig.suptitle(title, y=1.05)

        # Adjust layout
        plt.tight_layout()

        # Save plot if path provided
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
        hyperparameters: dict,
        window_idx: int,
    ) -> None:
        """
        Generate and log a time-series plot comparing predictions with actual data for a window.

        All required data (context/train/validate segments and predictions) must be provided
        by the caller. This function does not execute any models or perform data loading.
        """
        try:
            # Enforce 2D arrays with identical shapes
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
                    f"but got shape {timestamps_pred.shape}. Ensure timestamps_pred is 1D and matches the number of rows in y_pred."
                )

            num_targets = y_true.shape[1]

            # Create plots directory in current working directory
            plots_dir = Path.cwd() / "tensorboard" / "plots" / model_name
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Create subplots for each target
            fig, axes = plt.subplots(num_targets, 1, figsize=(15, 4 * num_targets))
            if num_targets == 1:
                axes = [axes]

            # Use prediction timestamps for both series; assume alignment
            time_axis = timestamps_pred

            for target_idx in range(num_targets):
                ax = axes[target_idx]  # type: ignore

                true_series = y_true[:, target_idx]
                pred_series = y_pred[:, target_idx]

                # Plot true values
                ax.plot(
                    time_axis,
                    true_series,
                    "b-",
                    label="True Values",
                    linewidth=2,
                    alpha=0.8,
                )

                # Plot predictions
                ax.plot(
                    time_axis,
                    pred_series,
                    "orange",
                    label="Predicted",
                    linewidth=2,
                    alpha=0.8,
                )

                # Customize subplot
                ax.set_title(
                    f"{model_name} - Target {target_idx + 1} (Window {window_idx})"
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Add hyperparameters info to the figure
            fig.suptitle(
                f"Best Hyperparameters: {hyperparameters}", fontsize=12, y=0.98
            )

            # Save plot
            plot_path = plots_dir / f"window_{window_idx}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Log to TensorBoard via LogManager
            LogManager.get_logger().log_image_file(
                image_path=str(plot_path),
                tag=f"{model_name}/forecast",
                step=window_idx,
            )

        except Exception as e:
            LogManager.get_logger().error(
                "Visualizer",
                f"Error generating forecast plot for {model_name}: {e}",
            )
