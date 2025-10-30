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

from ..config.configs import JobConfig
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

    def __init__(self, job_config: JobConfig):
        """
        Initialize the loader for a specific job configuration.

        Args:
            job_config: Aggregated configuration object that includes benchmark settings,
                dataset metadata, and preprocessing directives for the active task.
        """
        self.job_config = job_config
        self.config = job_config.benchmark_config
        self.task_config = job_config.task_config
        self.logger = job_config.logger
        self.preprocessor = Preprocessor(job_config)

    def _load_dataset(self, dataset_path: str) -> tuple:
        """
        Load a complete dataset file and extract basic metadata.

        Loads the entire CSV file into memory and extracts metadata from the first row.
        The CSV is expected to have columns: item_id, start, freq, and target.

        Args:
            dataset_path (str): Path to the CSV dataset file to load.

        Returns:
            tuple: A tuple containing (time_start, time_freq, target_raw) where:
                - time_start: Starting timestamp of the time series
                - time_freq: Frequency of the time series data
                - target_raw: Raw target data (will be processed by preprocessor)

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.

        Note:
            This method only extracts metadata and raw data. The actual data
            cleaning and preprocessing is handled by the Preprocessor class.
        """

        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load the csv data
        file_data = pd.read_csv(dataset_path, encoding="utf-8")

        # Extract basic information
        time_start = file_data["start"].iloc[0]
        time_freq = file_data["freq"].iloc[0]
        target_raw = file_data["target"].iloc[0]

        return time_start, time_freq, target_raw

    def generate_dataset_split(
        self, dataset_path: str, steps: list[tuple[str, int]], stride: int
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
        self.logger.debug("DataLoader", f"Extracting data from {dataset_path}")

        # Resolve actual dataset file path and load task-specific options
        dataset_file_path = dataset_path
        normalize = self.task_config.dataset.normalize
        handle_missing = self.task_config.dataset.handle_missing

        # All targets are 2D after cleaning: (n_steps, n_variates)
        timestamps, _, time_freq, target, scaler = self.preprocessor.clean(
            *self._load_dataset(str(dataset_file_path)), normalize, handle_missing
        )
        num_steps = target.shape[0]  # (n_steps, n_features): first dim is time-steps
        window_size = sum(seg_len for (_, seg_len) in steps)
        max_windows = self.config.evaluation.max_windows

        win = 0
        while win < max_windows:
            start = win * stride
            end = start + window_size
            if end > num_steps:
                break

            # Compute segment ranges for each step
            splits = {}
            for seg_name, seg_len in steps:
                end = start + seg_len
                splits[seg_name] = DatasetSplit(start=start, end=end)
                start = end

            # Construct the Dataset with dynamically assigned splits from steps
            window_kwargs = dict(
                timestamps=timestamps,
                target=target,
                scaler=scaler,
                metadata={
                    "dataset_path": str(dataset_file_path),
                    "window": win,
                    "freq": time_freq,
                },
            )
            # Include segment splits (e.g., context=..., train=..., validation=...)
            window_kwargs.update(splits)
            window = Dataset(**window_kwargs)

            yield win, window
            win += 1
