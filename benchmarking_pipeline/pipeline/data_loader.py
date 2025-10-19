"""
Data Loader for loading and preprocessing time series data chunks.

This module handles loading CSV data chunks and creating Dataset objects.
- Note: target columns are inferred from data
- Targets are kept as raw arrays, not converted to named columns.
- All data is treated as multivariate (univariate is just num_targets == 1)

The DataLoader automatically discovers the structure of time series data and creates
appropriate Dataset objects for training, validation, and testing.
"""
import os, sys, ast, csv
import pandas as pd
import numpy as np
import pdb
from typing import Optional, List, Dict, Any
from numpy.lib.stride_tricks import as_strided

from benchmarking_pipeline.utils.logger import Logger
from benchmarking_pipeline.utils.config_validator import load_config
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.pipeline.data_types import Dataset, DatasetSplit

class DataLoader:
    """
    Loads time series data chunks and creates Dataset objects.

    All data is treated as multivariate where univariate is just num_targets == 1.
    Targets are inferred from the data structure and kept as raw arrays.
    No artificial column naming is applied.

    The loader automatically splits data into train/validation/test sets based on
    the configured split ratios and handles different data formats.
    """
    def __init__(self, config_path: str, datasets_dir: str, run_dir: str):
        """
        Initialize DataLoader.

        Args:
            config: Configuration dictionary containing dataset parameters
                - dataset.name: Name of the dataset directory
                - dataset.split_ratio: List of train/val/test split ratios
        """
        self.config = load_config(config_path)
        self.logger = Logger(logs_dir="logs", name="DataLoader")
        self.datasets_dir = datasets_dir # ./datasets
        self.run_dir = run_dir
        self.preprocessor = Preprocessor(self.config)
        self.dataset_paths = self._load_dataset_paths()

    def _load_dataset_paths(self) -> List[str]:
        """
        Load dataset paths based on the dataset configuration.

        Returns:
            List of dataset file paths (CSV files) matching the configuration.

        Raises:
            ValueError: If no valid CSV files are found or an invalid path is specified.
        """
        logging = self.logger is not None
        dataset_name = self.config['task']['dataset']['name']

        if dataset_name == "*":  # wildcard to select all
            dataset_paths = [os.path.join(root, f) for root, _, files in os.walk(self.datasets_dir) for f in files if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in any subdirectory of {self.datasets_dir}")
        elif dataset_name.endswith("/*"):  # supports subdirectory (e.g., univariate/*)
            subdir_path = os.path.join(self.datasets_dir, dataset_name[:-2])
            if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
                raise ValueError(f"Subdirectory {subdir_path} does not exist or is not a directory")
            dataset_paths = [os.path.join(root, f) for root, _, files in os.walk(subdir_path) for f in files if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in {subdir_path}")
        else:
            dataset_dir_path = os.path.join(self.datasets_dir, dataset_name)
            if not os.path.exists(dataset_dir_path) or not os.path.isdir(dataset_dir_path):
                raise ValueError(f"Dataset in path {dataset_dir_path} does not exist or is not a directory")
            dataset_paths = [os.path.join(dataset_dir_path, f) for f in os.listdir(dataset_dir_path) if f.endswith(".csv")]
            if not dataset_paths:
                raise ValueError(f"No CSV files found in dataset directory {dataset_dir_path}")

        if logging: self.logger.debug(f"dataset_paths: {dataset_paths}")
        return dataset_paths

    def _load_dataset(self, dataset_path):

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load the chunk data
        file_data = pd.read_csv(dataset_path)

        # Extract basic information
        item_id = file_data["item_id"].iloc[0]
        time_start = file_data["start"].iloc[0]
        time_freq = file_data["freq"].iloc[0]

        # Handle targets by inference
        target = np.array(ast.literal_eval(file_data["target"].iloc[0]))
        return time_start, time_freq, target

    def generate_dataset_split(self, dataset_path: str, context_steps: int, train_steps: int, validate_steps: int):
        """
        Generate rolling windows for context, train, and validate splits, where each window starts validate_steps after the previous, so that windows do not overlap at all (stride = validate_steps). We stop when the end of the window for the next roll would exceed num_steps.

        For all slices extracted below (target[:, ...] or target[...]), these are numpy views,
        not copies: they share memory with the original array. No new memory is allocated
        except for metadata containers and timestamps.

        The progression is:
        [[----context----][---train----][---validate----]--------------------------------------]
        [----------------[-----context----][---train----][---validate----]---------------------]
        [---------------------------------[-----context----][---train----][---validate----]----]

        Args:
            context_steps: int, number of steps for context
            train_steps: int, number of steps for train
            validate_steps: int, number of steps for validation

        Yields:
            Dataset (with .context, .train, .validation, test=None)
        """

        self.logger.debug(f"Extracting data from {dataset_path}")
        # All targets are 2D after cleaning: (n_targets, n_steps)
        timestamps, time_start, time_freq, target = self.preprocessor.clean(*self._load_dataset(dataset_path))

        num_steps = target.shape[0]
        window_size = context_steps + train_steps + validate_steps
        stride = validate_steps  # advance by validate_steps every window

        # Advance by step size = validate_steps each time, extract "view" slices
        win = 0
        while True:
            start = win * stride
            end = start + window_size
            if end > num_steps: break

            ctx_start = start; ctx_end = ctx_start + context_steps
            train_start = ctx_end; train_end = train_start + train_steps
            val_start = train_end; val_end = val_start + validate_steps

            window = Dataset(
                timestamps=timestamps,
                target=target,
                context=DatasetSplit(
                    start=ctx_start,
                    end=ctx_end
                ),
                train=DatasetSplit(
                    start=train_start,
                    end=train_end
                ),
                validation=DatasetSplit(
                    start=val_start,
                    end=val_end
                ),
                metadata={
                    "dataset_path": dataset_path,
                    "window": win,
                    "freq": time_freq
                }
            )

            yield win, window
            win += 1
