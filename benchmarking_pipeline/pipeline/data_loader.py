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

from typing import Optional, List, Dict, Any
from numpy.lib.stride_tricks import as_strided

from benchmarking_pipeline.utils.logger import Logger
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
    def __init__(self, config: Dict[str, Any], datasets_dir: os.PathLike, run_dir: os.PathLike):
        """
        Initialize DataLoader.

        Args:
            config: Configuration dictionary containing dataset parameters
                - dataset.name: Name of the dataset directory
                - dataset.split_ratio: List of train/val/test split ratios
        """
        self.config = config
        self.logger = Logger(log_dir='logs', name='DataLoader')
        self.datasets_dir = datasets_dir # ./datasets
        self.run_dir = run_dir
        self.preprocessor = Preprocessor(config)
        # Create split dirs
        self.splits_dir = os.path.join(self.run_dir, "splits")
        dirs_to_create = [self.splits_dir]
        dirs_to_create += [os.path.join(self.splits_dir, subdir) for subdir in ["context", "train", "validate", "test"]]
        for d in dirs_to_create: os.makedirs(d, exist_ok=True)
        # Extract dataset paths
        self.dataset_paths = self._load_dataset_paths()

    def _load_dataset_paths(self) -> List[str]:
        """
        Load dataset paths based on the dataset configuration.

        Returns:
            List of dataset file paths (CSV files) matching the configuration.

        Raises:
            ValueError: If no valid CSV files are found or an invalid path is specified.
        """
        dataset_name = self.config['name']

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

        self.logger.debug(f"dataset_paths: {dataset_paths}")
        self.dataset_paths = dataset_paths

    def _load_dataset_batch(self, dataset_path, max_batch_size_mb=100):
        """
        Yields batches of targets from a csv file. Each batch contains at least 1 row and as many
        rows as can fit into < max_batch_size_mb MB RAM.
        Yields list-of-targets (batch).

        Args:
            dataset_path: Path to csv file (one row per item, with columns ... 'target')
            max_batch_size_mb: Max memory size per batch in MB (default 100)

        Yields:
            List of [target1, ..., targetN] for each batch
        """

        with open(dataset_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            # Read first row to estimate memory size per row
            try:
                first_row = next(reader)
            except StopIteration:
                raise ValueError(f"CSV file '{dataset_path}' is empty (no data rows found).")

            target = ast.literal_eval(first_row[headers.index("target")])
            row_bytes = sys.getsizeof(first_row)

            batch_rows = max(1, int((max_batch_size_mb * 1024 * 1024) // row_bytes))

            # Prepare first batch
            targets_batch = [target]
            rows_in_batch = 1
            for row in reader:
                target = ast.literal_eval(row[headers.index("target")])
                targets_batch.append(target)
                rows_in_batch += 1
                if rows_in_batch >= batch_rows:
                    yield targets_batch
                    targets_batch = []
                    rows_in_batch = 0
            if targets_batch:
                yield targets_batch

    def generate_dataset_split(self, context_steps: int, train_steps: int, validate_steps: int, test_steps: int = 0):
        window_size = context_steps + train_steps + validate_steps + test_steps

        # Compute indices
        ctx_idx = context_steps
        trn_idx = ctx_idx + train_steps
        vld_idx = trn_idx + validate_steps
        tst_idx = vld_idx + test_steps

        for dataset_path in self.dataset_paths:
            row_idx = 0
            for batch_idx, batch in enumerate(self._load_dataset_batch(dataset_path)):
                for target in batch:
                    target = np.array(target)
                    target = self.preprocessor.clean(target)

                    num_steps = len(target) if target.ndim == 1 else len(target[0])
                    if num_steps < window_size:
                        raise ValueError(f"Error in file '{dataset_path}', batch row {row_idx}: num_steps ({num_steps}) exceeds window_size ({window_size})")

                    stride = target.strides[0]
                    num_windows = num_steps - window_size + 1

                    # Verify we only support 1D or 2D (multivariate timeseries)
                    if target.ndim == 1: # Univariate case
                        shape = (num_windows, window_size)
                        windows = as_strided(
                            target,
                            shape=shape,
                            strides=(stride, stride),
                            writeable=False,
                        )
                    elif target.ndim == 2:
                        # Multivariate: (num_targets, num_steps) --> stride along axis 1
                        shape = (num_windows, target.shape[0], window_size)
                        windows = as_strided(
                            target,
                            shape=shape,
                            strides=(target.strides[1], target.strides[0], target.strides[1]),
                            writeable=False,
                        )
                    else:
                        raise ValueError("target array must be 1D or 2D")

                    # Now, make a single slice operation for all windows & yield
                    for win in range(num_windows):
                        win_slice = windows[win]
                        if target.ndim == 2:
                            # Convert from (targets, window_size)
                            window_data = win_slice
                        else:
                            window_data = win_slice  # shape (window_size,)

                        # Efficient slicing: avoid redundant slicing and allow view semantics
                        if target.ndim == 2:
                            context_set = window_data[:, :ctx_idx]
                            train_set = window_data[:, ctx_idx:trn_idx]
                            validate_set = window_data[:, trn_idx:vld_idx]
                            test_set = window_data[:, vld_idx:tst_idx]
                        else:
                            context_set = window_data[:ctx_idx]
                            train_set = window_data[ctx_idx:trn_idx]
                            validate_set = window_data[trn_idx:vld_idx]
                            test_set = window_data[vld_idx:tst_idx]

                        # Wrap each set as a DatasetSplit (here only targets are set, features/timestamps/metadata are None)
                        dataset = Dataset(
                            context = DatasetSplit(targets=context_set),
                            train = DatasetSplit(targets=train_set),
                            validation = DatasetSplit(targets=validate_set),
                            test = DatasetSplit(targets=test_set),
                            metadata = {
                                "file_path": dataset_path,
                                "batch": batch_idx,
                                "row": row_idx,
                                "window": win
                            }
                        )

                        row_idx += 1
                        yield dataset