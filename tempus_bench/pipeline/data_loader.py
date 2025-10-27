"""
Data Loader for loading and preprocessing time series data chunks.

This module provides the DataLoader class for handling time series data in CSV format.
It automatically discovers dataset structure, loads data chunks, and creates Dataset objects
for machine learning workflows.

Key Features:
    - Automatic target column inference from data structure
    - Support for both univariate and multivariate time series
    - Rolling window generation with configurable splits
    - Flexible dataset path resolution (wildcards, subdirectories)
    - Integration with preprocessing pipeline

The DataLoader treats all data as multivariate where univariate is simply num_targets == 1.
Targets are kept as raw arrays without artificial column naming for maximum flexibility.

Example:
    >>> loader = DataLoader(config_path="config.yaml", tasks_dir="./tasks", run_dir="./runs")
    >>> for window_idx, dataset in loader.generate_dataset_split(
    ...     dataset_path="data.csv",
    ...     steps=[('context', 24), ('train', 12), ('validate', 6)],
    ...     stride=1
    ... ):
    ...     # Process dataset...
"""
import pdb
import numpy as np
import pandas as pd
import os, sys, ast, csv
from pathlib import Path

from typing import Optional, List, Dict, Any

from tempus_bench.utils.logger import get_logger
from tempus_bench.utils.paths import get_tasks_dir
from tempus_bench.config import load_config
from tempus_bench.pipeline.preprocessor import Preprocessor
from tempus_bench.pipeline.data_types import Dataset, DatasetSplit

class DataLoader:
    """
    Loads and processes time series data chunks into Dataset objects.

    The DataLoader handles CSV-based time series data with automatic structure discovery.
    It supports flexible dataset path resolution and creates rolling windows for training.

    Attributes:
        config (dict): Configuration dictionary loaded from config_path
        logger (Logger): Logger instance for debugging and monitoring
        tasks_dir (str): Base directory containing task files
        run_dir (str): Directory for storing run outputs
        preprocessor (Preprocessor): Preprocessor instance for data cleaning
        dataset_paths (List[str]): List of discovered dataset file paths

    Note:
        All data is treated as multivariate where univariate is simply num_targets == 1.
        Targets are inferred from data structure and kept as raw arrays without
        artificial column naming for maximum flexibility.
    """
    def __init__(self, config_path: str, run_dir: str):
        """
        Initialize DataLoader with configuration and directory paths.

        Args:
            config_path (str): Path to the configuration YAML file containing
                dataset parameters and task specifications.
            run_dir (str): Directory for storing run outputs and logs.

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: If no valid dataset paths are found.

        Note:
            The configuration should contain:
            - task.dataset.name: Dataset name or pattern (supports wildcards)
            - task.max_windows: Maximum number of windows to generate
            - Other preprocessing and model parameters
        """
        self.config = load_config(config_path)
        self.logger = get_logger(os.path.join(run_dir, 'logs'))
        self.tasks_dir = get_tasks_dir()
        self.run_dir = run_dir
        self.preprocessor = Preprocessor(self.config, task_path=None)  # Will be set per task
        self.dataset_paths = self._load_dataset_paths()

    def _load_dataset_paths(self) -> List[str]:
        """
        Discover and load dataset file paths based on configuration.

        Supports multiple dataset selection patterns:
        - Specific dataset: "dataset_name" -> ./datasets/dataset_name/*.csv
        - Subdirectory wildcard: "univariate/*" -> ./datasets/univariate/*.csv
        - Global wildcard: "*" -> ./datasets/**/*.csv

        Returns:
            List[str]: List of absolute paths to CSV files matching the pattern.

        Raises:
            ValueError: If no valid CSV files are found or specified path doesn't exist.

        Note:
            All discovered paths are logged at debug level for troubleshooting.
        """
        logging = self.logger is not None
        dataset_name = self.config['task']['path']

        if dataset_name == "*":  # wildcard to select all
            dataset_paths = [os.path.join(root, f) for root, _, files in os.walk(self.tasks_dir) for f in files if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in any subdirectory of {self.tasks_dir}")
        elif dataset_name.endswith("/*"):  # supports subdirectory (e.g., univariate/*)
            subdir_path = os.path.join(self.tasks_dir, dataset_name[:-2])
            if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
                raise ValueError(f"Subdirectory {subdir_path} does not exist or is not a directory")
            dataset_paths = [os.path.join(root, f) for root, _, files in os.walk(subdir_path) for f in files if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in {subdir_path}")
        else:
            dataset_dir_path = os.path.join(self.tasks_dir, dataset_name)
            if not os.path.exists(dataset_dir_path) or not os.path.isdir(dataset_dir_path):
                raise ValueError(f"Dataset in path {dataset_dir_path} does not exist or is not a directory")
            dataset_paths = [os.path.join(dataset_dir_path, f) for f in os.listdir(dataset_dir_path) if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in dataset directory {dataset_dir_path}")

        if logging: self.logger.debug("DataLoader", f"dataset_paths: {dataset_paths}")
        return dataset_paths
    
    def _get_task_path_from_dataset_path(self, dataset_path: str) -> Optional[str]:
        """Get the task directory path from a dataset file path."""
        dataset_file = Path(dataset_path)
        # The task directory is the parent of the CSV file
        task_dir = dataset_file.parent
        return str(task_dir)

    def _load_dataset(self, dataset_path: str) -> tuple:
        """
        Load a single dataset file and extract basic metadata.

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

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load the chunk data
        file_data = pd.read_csv(dataset_path)

        # Extract basic information
        item_id = file_data["item_id"].iloc[0]
        time_start = file_data["start"].iloc[0]
        time_freq = file_data["freq"].iloc[0]

        # Handle targets by inference - return raw data for preprocessor to clean
        target_raw = file_data["target"].iloc[0]

        return time_start, time_freq, target_raw

    def generate_dataset_split(self, dataset_path: str, steps: list[tuple[str, int]], stride: int):
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

        Note:
            - Windows are limited by max_windows configuration parameter
            - Each Dataset includes timestamps, target data, scaler, and metadata
            - Target data is preprocessed and normalized by the Preprocessor
            - Metadata includes dataset_path, window index, and frequency

        Example:
            >>> for win_idx, dataset in loader.generate_dataset_split(
            ...     "data.csv",
            ...     [('context', 24), ('train', 12), ('validate', 6)],
            ...     stride=1
            ... ):
            ...     context_data = dataset.context.target  # Context segment
            ...     train_data = dataset.train.target     # Training segment
            ...     val_data = dataset.validate.target    # Validation segment
        """
        self.logger.debug("DataLoader", f"Extracting data from {dataset_path}")
        
        # Determine task path from dataset path
        task_path = self._get_task_path_from_dataset_path(dataset_path)
        
        # Create task-specific preprocessor
        task_preprocessor = Preprocessor(self.config, task_path)
        
        # All targets are 2D after cleaning: (n_steps, n_variates)
        timestamps, time_start, time_freq, target, scaler = task_preprocessor.clean(*self._load_dataset(dataset_path))
        num_steps = target.shape[0]  # (n_steps, n_features): first dim is time-steps
        window_size = sum(seg_len for (_, seg_len) in steps)
        max_windows = self.config['evaluation']['max_windows']

        win = 0
        while win < max_windows:
            start = win * stride
            end = start + window_size
            if end > num_steps: break

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
                    "dataset_path": dataset_path,
                    "window": win,
                    "freq": time_freq
                }
            )
            # Include segment splits (e.g., context=..., train=..., validation=...)
            window_kwargs.update(splits)
            window = Dataset(**window_kwargs)

            yield win, window
            win += 1
