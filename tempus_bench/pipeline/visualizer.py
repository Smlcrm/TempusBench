"""
Visualization utilities for the benchmarking pipeline.
"""

from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..utils.log_manager import LogManager


class Visualizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer with configuration.

        Args:
            config (dict, optional): Configuration dictionary with visualization parameters.
        """
        self.config = config if config is not None else {}
        # Use a built-in style instead of seaborn
        plt.style.use("fivethirtyeight")
        # Set seaborn style separately
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

        Args:
            y_true: Actual values
            y_pred: Predicted values
            time_index: Optional time index for x-axis
            y_pred_lower: Lower bound of prediction interval (optional)
            y_pred_upper: Upper bound of prediction interval (optional)
            title: Plot title
            save_path: If provided, save the plot to this path
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

        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: If provided, save the plot to this path
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
