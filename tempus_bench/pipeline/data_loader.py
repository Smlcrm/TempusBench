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
import pandas as pd

from pathlib import Path

from tempus_bench.config.config import ConfigAdapterMixin
from tempus_bench.config.models import JobConfig
from tempus_bench.pipeline.preprocessor import Preprocessor
from tempus_bench.pipeline.data_types import Dataset, DatasetSplit

class DataLoader(ConfigAdapterMixin):
    """
    Loads and processes complete time series datasets into Dataset objects.

    The DataLoader handles CSV-based time series data, loading entire datasets into memory
    and creating rolling windows for training. It works with a single dataset at a time,
    suitable for the task-based execution model.

    Attributes:
        config (dict): Configuration dictionary loaded from config_path
        logger (Logger): Logger instance for debugging and monitoring
        tasks_dir (Path): Base directory containing task files
        run_path (str): Directory for storing run outputs

    Note:
        All data is treated as multivariate where univariate is simply num_targets == 1.
        Targets are inferred from data structure and kept as raw arrays without
        artificial column naming for maximum flexibility.
    """
    def __init__(self, config: JobConfig):
        """
        Initialize DataLoader with configuration and directory paths.

        Args:
            config_path (str): Path to the configuration YAML file containing
                dataset parameters and task specifications.
            logs_path (str): Directory for storing log files.

        Raises:
            FileNotFoundError: If config_path does not exist.

        Note:
            The configuration should contain:
            - evaluation.max_windows: Maximum number of windows to generate
            - Other preprocessing and model parameters
        """
        self.preprocessor = Preprocessor(config)

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
        file_data = pd.read_csv(dataset_path, encoding='utf-8')

        # Extract basic information
        time_start = file_data["start"].iloc[0]
        time_freq = file_data["freq"].iloc[0]
        target_raw = file_data["target"].iloc[0]

        return time_start, time_freq, target_raw

    def _get_task_path_from_dataset_path(self, dataset_path: str) -> str:
        """
        Extract task directory path from dataset CSV file path.
        
        Args:
            dataset_path: Path to the CSV dataset file
            
        Returns:
            Path to the task directory containing the dataset file
        """
        dataset_file = Path(dataset_path)
        return str(dataset_file.parent)

    def _load_task_config(self, task_path: str) -> tuple[bool, str]:
        """
        Load task-specific configuration from task.yaml (uses first task definition).
        
        Args:
            task_path: Path to the task directory containing task.yaml
            
        Returns:
            Tuple of (normalize, handle_missing) with defaults (True, 'interpolate')
        """
        import yaml
        
        task_dir = Path(task_path)
        task_config_path = task_dir / "task.yaml"

        if not task_config_path.exists():
            self.logger.warning("DataLoader", f"Task config file not found: {task_config_path}, using defaults")
            return True, 'interpolate'

        try:
            with open(task_config_path, 'r') as f:
                # Load all YAML documents from the file
                task_configs = list(yaml.safe_load_all(f))

            # Use the first valid task configuration
            for task_config in task_configs:
                if task_config is None or 'task' not in task_config:
                    continue

                task_data = task_config['task']
                if 'dataset' not in task_data:
                    continue

                # Extract dataset configuration
                dataset_config = task_data.get('dataset', {})
                normalize = dataset_config.get('normalize', True)
                handle_missing = dataset_config.get('handle_missing', 'interpolate')
                return normalize, handle_missing

            # If no valid task config found, use defaults
            self.logger.warning(f"No valid task configuration found in {task_config_path}, using defaults")
            return True, 'interpolate'

        except Exception as e:
            self.logger.warning(f"Failed to load task config from {task_config_path}: {e}, using defaults")
            return True, 'interpolate'

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

        # Load task-specific configuration
        normalize, handle_missing = self._load_task_config(task_path)

        # All targets are 2D after cleaning: (n_steps, n_variates)
        timestamps, _, time_freq, target, scaler = self.preprocessor.clean(*self._load_dataset(dataset_path), normalize, handle_missing)
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
