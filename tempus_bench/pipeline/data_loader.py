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
import json

import pandas as pd
import numpy as np

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
        The CSV is expected to have either:
        1. Standard format: columns item_id, start, freq, and target (univariate/multivariate)
        2. Covariate format: columns variable_name, timestamps, values, and variable_type

        The data is then preprocessed using the Preprocessor class and stored as a Dataset
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

        # Check if this is a covariate dataset
        if set(file_data.columns) == {"variable_name", "timestamps", "values", "variable_type"}:
            self._load_covariate_dataset(file_data)
        else:
            # Standard univariate/multivariate dataset
            self._load_standard_dataset(file_data)

    def _load_standard_dataset(self, file_data: pd.DataFrame):
        """
        Load a standard univariate/multivariate dataset.

        Args:
            file_data: DataFrame with columns item_id, start, freq, target
        """
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
            covariate=None,
            metadata={
                "dataset_path": str(self.dataset_path),
                "time_start": time_start,
                "time_freq": time_freq,
            },
        )

    def _load_covariate_dataset(self, file_data: pd.DataFrame):
        """
        Load a covariate dataset with separate target and covariate variables.

        Args:
            file_data: DataFrame with columns variable_name, timestamps, values, variable_type

        The variable_type column indicates whether each row is a 'target' or 'covariate'.
        All variables should have the same timestamps.
        """
        target_rows = file_data[file_data["variable_type"] == "target"]
        covariate_rows = file_data[file_data["variable_type"] == "covariate"]

        if len(target_rows) == 0:
            raise ValueError("Covariate dataset must have at least one 'target' variable")

        # Parse the first target row to get timestamps and frequency info
        first_target = target_rows.iloc[0]
        timestamps_str = json.loads(first_target["timestamps"])
        timestamps = pd.to_datetime(timestamps_str)

        # Infer frequency
        if len(timestamps) > 1:
            time_freq = pd.infer_freq(timestamps)
        else:
            time_freq = "D"  # Default to daily

        time_start = str(timestamps[0])
        

        # Parse all target variables
        target_data = []
        for _, row in target_rows.iterrows():
            values = json.loads(row["values"])
            target_data.append(values)

        # target_data: list of [var1_series, var2_series, var3_series] - each series has num_steps values
        # Preprocessor expects [[var1_series], [var2_series], ...] (list of variates, each variate = full time series)
        target_raw_str = str(target_data)

        # Parse all covariate variables (same format for preprocessor)
        covariate_data = []
        for _, row in covariate_rows.iterrows():
            values = json.loads(row["values"])
            covariate_data.append(values)

        # Apply preprocessing (normalization, handling missing values)
        normalize = self.task_config.dataset.normalize
        handle_missing = self.task_config.dataset.handle_missing

        preprocessor = Preprocessor(self.task_config, self.evaluation_config)

        # Process target data - preprocessor expects list of variates (each variate = full time series)
        #print("time_start, time_freq, target_raw_str, normalize, handle_missing",time_start, time_freq, target_raw_str, normalize, handle_missing)
        timestamps, time_start, time_freq, target, scaler = preprocessor.clean(
            time_start, time_freq, target_raw_str, normalize, handle_missing
        )

        # Process covariate data if present - preprocessor expects list of variates (each = full time series)
        if covariate_data:
            covariate_raw_str = str(covariate_data)
            _, _, _, covariate, _ = preprocessor.clean(
                time_start, time_freq, covariate_raw_str, normalize=False, handle_missing=handle_missing
            )
        else:
            covariate = None

        self.scaler = scaler

        assert target.ndim == 2  # (num_steps, num_targets)
        assert timestamps.ndim == 1  # (num_steps,)
        assert covariate.ndim == 2  # (num_steps, num_covariates)

        # Construct the Dataset with covariates
        self.dataset = Dataset(
            timestamps=timestamps.tolist(),
            target=target.tolist(),
            covariate=covariate.tolist(),
            metadata={
                "dataset_path": str(self.dataset_path),
                "time_start": time_start,
                "time_freq": time_freq,
                "num_targets": target.shape[1],
                "num_covariates": covariate.shape[1],
            },
        )
