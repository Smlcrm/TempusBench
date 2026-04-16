"""
Core data types used throughout the benchmarking pipeline.

This module defines the core data structures used for representing time series
datasets, dataset splits, and task results in the benchmarking pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, List


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
    values, optional covariates, and metadata.

    Attributes:
        timestamps (np.ndarray): Array of timestamps of shape (num_steps,).
        target (np.ndarray): 2D numpy array of target values with shape
            (num_steps, num_targets).
        covariate (Optional[np.ndarray]): 2D numpy array of covariate values with shape
            (num_steps, num_covariates). Set to None if no covariates are present.
        scaler (Optional[StandardScaler]): Scaler used for normalization
            (if any). Set to None if no normalization was applied.
        metadata (Optional[Dict[str, Any]]): Dictionary of metadata including
            frequency, start time, and other dataset properties.
    """

    timestamps: List[float]  # Array of timestamps (same length as num_steps) or None
    target: List[List[float]]  # 2D np.ndarray of Target values
    covariate: Optional[List[List[float]]] = None  # 2D array of Covariate values
    metadata: Optional[Dict[str, Any]] = None

    def generate_dataset_split(
        self, steps: list[tuple[str, int]], stride: int, max_windows: int
    ):
        """
        Generate rolling windows over a time series with configurable segments.

        Creates sliding windows over the time series data, where each window is split
        into multiple segments (e.g., context, train, validation) as specified.

        Args:
            dataset_path (str): Path to the CSV dataset file to process.
            steps (list[tuple[str, int]]): List of (segment_name, num_steps) tuples
                defining how to split each window. Example:
                [('context', 24), ('train', 12), ('validate', 6)]
            stride (int): Number of time steps to advance between consecutive windows.
                stride=1 creates overlapping windows, stride=window_size creates non-overlapping.

        Yields:
            tuple[int, Dataset]: Generator yielding (window_index, dataset) pairs where:
                - window_index (int): Zero-based index of the current window
                - dataset (Dataset): Dataset object containing the window data with
                    segment splits (context, train, validation, etc.) and metadata
                stride=1 creates overlapping windows, stride=window_size creates non-overlapping.

        Notes:
            - Windows are limited by the `evaluation.max_windows` configuration parameter.
            - Each yielded `Dataset` includes timestamps, target data, scaler, and metadata.
            - Target data is preprocessed and normalized by the `Preprocessor`.
        """

        # Resolve actual dataset file path and load task-specific options

        num_steps = len(self.target)
        window_size = sum(seg_len for (_, seg_len) in steps)

        window_idx = 0
        while window_idx < max_windows:
            start = window_idx * stride
            end = start + window_size
            if end > num_steps:
                break

            # Compute segment ranges for each step
            splits = {}
            for seg_name, seg_len in steps:
                end = start + seg_len
                splits[seg_name] = DatasetSplit(start=start, end=end)
                start = end

            window_idx += 1
            yield splits


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
