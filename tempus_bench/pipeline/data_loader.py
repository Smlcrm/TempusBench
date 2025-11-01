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
    >>> loader = DataLoader(task_config, evaluation_config)
    >>> for window_idx, dataset in loader.generate_dataset_split(
    ...     steps=[('context', 24), ('train', 12), ('validate', 6)],
    ...     stride=1
    ... ):
    ...     # Process dataset...
"""

from pathlib import Path

import pandas as pd

from ..utils.configs import EvaluationConfig, TaskConfig
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

    def __init__(self, task_config: TaskConfig, evaluation_config: EvaluationConfig):
        """
        Initialize the loader for a specific task and evaluation configuration.

        Args:
            task_config: Task configuration object that includes dataset metadata
                and preprocessing directives for the active task.
            evaluation_config: Evaluation configuration object that includes
                benchmark settings for evaluation.
        """
        self.task_config = task_config
        self.evaluation_config = evaluation_config
        task_path = Path(self.task_config.task_path)
        self.dataset_path = task_path / self.task_config.dataset.file_name

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

        preprocessor = Preprocessor(self.task_config, self.evaluation_config)
        # All targets are 2D after cleaning: (n_steps, n_variates)
        timestamps, time_start, time_freq, target, scaler = preprocessor.clean(
            time_start, time_freq, target_raw, normalize, handle_missing
        )

        self.scaler = scaler

        assert target.ndim == 2  # (num_steps, num_targets)
        assert timestamps.ndim == 1  # (num_steps,)

        # Construct the Dataset with dynamically assigned splits from steps
        self.dataset = Dataset(
            timestamps=timestamps.tolist(),
            target=target.tolist(),
            metadata={
                "dataset_path": str(self.dataset_path),
                "time_start": time_start,
                "time_freq": time_freq,
            },
        )
