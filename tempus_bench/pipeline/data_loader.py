"""
Data Loader for loading and preprocessing time series data.

This module provides the DataLoader class for handling time series data in CSV format.
It loads complete dataset files and creates Dataset objects for machine learning workflows.

Key Features:
    - Complete CSV dataset loading (not chunked)
    - Automatic target column inference from data structure
    - Support for both univariate and multivariate time series
    - Rolling window generation with configurable splits
    - Integration with preprocessing pipeline

The DataLoader treats all data as multivariate where univariate is simply num_targets == 1.
Targets are kept as raw arrays without artificial column naming for maximum flexibility.

Example:
    >>> loader = DataLoader(config_path="config.yaml", run_path="./runs")
    >>> for window_idx, dataset in loader.generate_dataset_split(
    ...     dataset_path="data.csv",
    ...     steps=[('context', 24), ('train', 12), ('validate', 6)],
    ...     stride=1
    ... ):
    ...     # Process dataset...
"""

from pathlib import Path

import pandas as pd

from ..utils.configs import JobConfig
from ..utils.logger import LoggerManager
from .data_types import Dataset, DatasetSplit
from .preprocessor import Preprocessor


class DataLoader:
    """
    Loads and processes complete time series datasets into Dataset objects.

    The DataLoader handles CSV-based time series data, loading entire datasets into memory
    and creating rolling windows for training. It works with a single dataset at a time,
    suitable for the task-based execution model.

    All data is treated as multivariate where univariate is simply num_targets == 1.
    Targets are inferred from data structure and kept as raw arrays without
    artificial column naming for maximum flexibility.
    """

    def __init__(self, job_config: JobConfig, logger: LoggerManager):
        """
        Initialize the loader for a specific job configuration.

        Args:
            job_config: Aggregated configuration object that includes benchmark settings,
                dataset metadata, and preprocessing directives for the active task.
            logger: Logger instance for logging.
        """
        self.job_config = job_config
        self.evaluation_config = job_config.evaluation_config
        self.task_config = job_config.task_config
        task_path = Path(self.task_config.task_path)
        self.dataset_path = task_path / self.task_config.dataset.file_name
        self.logger = logger

        self._load_dataset()

    def _load_dataset(self):
        """
        Load a complete dataset file and extract basic metadata.

        Loads the entire CSV file into memory and extracts metadata from the first row.
        The CSV is expected to have columns: item_id, start, freq, and target. The
        data is then preprocessed using the Preprocessor class and stored as a Dataset
        instance.

        Note:
            This method only extracts metadata and raw data. The actual data cleaning
            and preprocessing is handled by the Preprocessor class.

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            ValueError: If the dataset cannot be processed or has invalid format.
        """

        # Load the csv data
        file_data = pd.read_csv(self.dataset_path, encoding="utf-8")

        # Extract basic information
        time_start = file_data["start"].iloc[0]
        time_freq = file_data["freq"].iloc[0]
        target_raw = file_data["target"].iloc[0]

        normalize = self.task_config.dataset.normalize
        handle_missing = self.task_config.dataset.handle_missing

        preprocessor = Preprocessor(self.job_config, self.logger)
        # All targets are 2D after cleaning: (n_steps, n_variates)
        timestamps, time_start, time_freq, target, scaler = preprocessor.clean(
            time_start, time_freq, target_raw, normalize, handle_missing
        )

        # Construct the Dataset with dynamically assigned splits from steps
        self.dataset = Dataset(
            timestamps=timestamps,
            target=target,
            scaler=scaler,
            metadata={
                "dataset_path": str(self.dataset_path),
                "time_start": time_start,
                "time_freq": time_freq,
            },
        )

    def generate_dataset_split(
        self, steps: list[tuple[str, int]], stride: int
    ) -> dict:
        """
        Generate rolling windows over a time series with configurable segments.

        Creates sliding windows over the time series data, where each window is split
        into multiple segments (e.g., context, train, validation) as specified. Windows
        are generated with the specified stride until the maximum number of windows is
        reached or the end of the time series is reached.

        Args:
            steps (list[tuple[str, int]]): List of (segment_name, num_steps) tuples
                defining how to split each window. Example:
                [('context', 24), ('train', 12), ('validate', 6)]
            stride (int): Number of time steps to advance between consecutive windows.
                stride=1 creates overlapping windows, stride=window_size creates non-overlapping.

        Yields:
            dict: Dictionary mapping segment names to DatasetSplit objects. Each
                DatasetSplit contains start and end indices for that segment in the
                window. Example: {'context': DatasetSplit(start=0, end=24), ...}

        Notes:
            - Windows are limited by the `evaluation.max_windows` configuration parameter.
            - Each yielded dictionary contains DatasetSplit objects for all segments.
            - Segment indices are computed relative to the full dataset, not the window.
        """
        self.logger.debug("DataLoader", f"Extracting data from {self.dataset_path}")

        # Resolve actual dataset file path and load task-specific options

        num_steps = self.dataset.target.shape[
            0
        ]  # (n_steps, n_features): first dim is time-steps
        window_size = sum(seg_len for (_, seg_len) in steps)
        max_windows = self.evaluation_config.max_windows

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
