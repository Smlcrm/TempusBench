"""
Core data types used throughout the benchmarking pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    """
    Represents a dataset split (train/val/test/etc) for time series, as actually produced by DataLoader.
    """

    start: int
    end: int
    metadata: Optional[Dict[str, Any]] = None  # Arbitrary additional per-split metadata


@dataclass
class Dataset:
    """
    Container for all dataset splits.
    """

    timestamps: np.ndarray  # Array of timestamps (same length as num_steps) or None
    target: np.ndarray  # 2D np.ndarray of Target values
    scaler: Optional[StandardScaler] = None  # Scaler used for normalization (if any)
    metadata: Optional[Dict[str, Any]] = None

    def generate_dataset_split(self, steps: list[tuple[str, int]], stride: int, max_windows: int):
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

        num_steps = self.target.shape[0]  # (n_steps, n_features): first dim is time-steps
        
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

            yield splits

@dataclass
class TaskResult:
    """
    Contains results of a model task, including hyperparameter optimization and dataset metadata.
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
