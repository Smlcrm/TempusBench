"""
Core data types used throughout the benchmarking pipeline.

This module defines the core data structures used for representing time series
datasets, dataset splits, and task results in the benchmarking pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    """
    Represents a dataset split (context/train/validate) for time series.

    This class encapsulates the start and end indices of a dataset split as
    produced by DataLoader, along with optional metadata.

    Attributes:
        start (int): Starting index (inclusive) of the split.
        end (int): Ending index (exclusive) of the split.
        metadata (Optional[Dict[str, Any]]): Arbitrary additional per-split metadata.
    """

    start: int
    end: int
    metadata: Optional[Dict[str, Any]] = None  # Arbitrary additional per-split metadata


@dataclass
class Dataset:
    """
    Container for time series dataset with all splits and metadata.

    This class holds the complete time series data including timestamps, target
    values, optional scaler for normalization, and metadata.

    Attributes:
        timestamps (np.ndarray): Array of timestamps of shape (num_steps,).
        target (np.ndarray): 2D numpy array of target values with shape
            (num_steps, num_targets).
        scaler (Optional[StandardScaler]): Scaler used for normalization
            (if any). Set to None if no normalization was applied.
        metadata (Optional[Dict[str, Any]]): Dictionary of metadata including
            frequency, start time, and other dataset properties.
    """

    timestamps: np.ndarray  # Array of timestamps (same length as num_steps) or None
    target: np.ndarray  # 2D np.ndarray of Target values
    scaler: Optional[StandardScaler] = None  # Scaler used for normalization (if any)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """
    Contains results of a model task execution.

    This class stores the complete results of running a model on a task, including
    optimal hyperparameters, evaluation metrics, and dataset/task metadata.

    Attributes:
        optimal_hyperparameters (Dict[int, Dict[str, float]]): Dictionary mapping
            window index to optimal hyperparameter dictionary for that window.
        final_evaluations (Dict[str, float]): Dictionary mapping metric names to
            averaged evaluation scores across all windows.
        dataset_path (str): Path to the dataset file used for this task.
        context_window (int): Number of context steps used for training.
        forecast_horizon (int): Number of steps forecasted ahead.
        model_type (Literal["deterministic", "stochastic", "hybrid"]): Type of
            model used (deterministic for point forecasts, stochastic for probabilistic,
            hybrid for both).
        tuning_loss (str): Loss metric used for hyperparameter selection.
        dataset_normalize (bool): Whether the dataset was normalized.
        dataset_handle_missing (Literal[...]): Strategy used for handling missing values.
    """

    optimal_hyperparameters: Dict[int, Dict[str, float]]
    final_evaluations: Dict[str, float]
    dataset_path: str
    context_window: int
    forecast_horizon: int
    model_type: Literal["deterministic", "stochastic", "hybrid"]
    tuning_loss: str
    dataset_normalize: bool
    dataset_handle_missing: Literal[
        "interpolate", "mean", "median", "drop", "forward_fill", "backward_fill"
    ]
