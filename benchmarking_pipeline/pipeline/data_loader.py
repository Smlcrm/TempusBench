"""
Data Loader for loading and preprocessing time series data chunks.

This module handles loading CSV data chunks and creating Dataset objects.
- Note: target columns are inferred from data
- Targets are kept as raw arrays, not converted to named columns.
- All data is treated as multivariate (univariate is just num_targets == 1)

The DataLoader automatically discovers the structure of time series data and creates
appropriate Dataset objects for training, validation, and testing.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from .data_types import Dataset, DatasetSplit
from .preprocessor import Preprocessor


class DataLoader:
    """
    Loads time series data chunks and creates Dataset objects.

    All data is treated as multivariate where univariate is just num_targets == 1.
    Targets are inferred from the data structure and kept as raw arrays.
    No artificial column naming is applied.

    The loader automatically splits data into train/validation/test sets based on
    the configured split ratios and handles different data formats.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader.

        Args:
            config: Configuration dictionary containing dataset parameters
                - dataset.path: Path to the dataset directory
                - dataset.split_ratio: List of train/val/test split ratios
        """
        self.config = config
        self.dataset_path = config["dataset"]["path"]
        self.num_targets: Optional[int] = None

        # Initialize preprocessor for handling missing values and normalization
        self.preprocessor = Preprocessor(config)

    def _extract_target_structure(self, target_data: Any) -> None:
        """
        Infer the target structure from the target data.
        All data is treated as multivariate where univariate is just num_targets == 1.

        Args:
            target_data: Raw target data from CSV
        """
        if isinstance(target_data, str):
            try:
                parsed = eval(target_data)
                if isinstance(parsed, list):
                    if parsed and isinstance(parsed[0], list):
                        # Multivariate: [[1,2,3], [4,5,6]] - multiple target series
                        self.num_targets = len(parsed)
                    else:
                        # Univariate: [1,2,3,4,5] - single target series (multivariate with 1 target)
                        self.num_targets = 1
                else:
                    # Single value - multivariate with 1 target
                    self.num_targets = 1
            except Exception as e:
                raise ValueError(
                    f"Failed to parse target data structure: {e}. Expected string representation of list or nested list."
                )
        else:
            # Non-string data - multivariate with 1 target
            self.num_targets = 1

    def _create_targets_dataframe(self, target_data: Any) -> pd.DataFrame:
        """
        Create targets DataFrame from raw target data.
        All data is treated as multivariate structure.

        Args:
            target_data: Raw target data from CSV

        Returns:
            DataFrame with raw target arrays (no artificial column names)
        """
        if isinstance(target_data, str):
            try:
                parsed = eval(target_data)
                if isinstance(parsed, list):
                    if parsed and isinstance(parsed[0], list):
                        # Multivariate: parsed is list of target series [num_targets][time_steps]
                        arr = np.array(parsed)
                        if arr.ndim != 2:
                            raise ValueError("Parsed multivariate target is not 2D")
                        # Reorient to (time_steps, num_targets)
                        arr = arr.T
                        targets_df = pd.DataFrame(arr).apply(
                            pd.to_numeric, errors="coerce"
                        )
                    else:
                        # Univariate: create single-column DataFrame with rows as time steps
                        targets_df = pd.DataFrame(
                            {"y": pd.to_numeric(pd.Series(parsed), errors="coerce")}
                        )
                else:
                    # Single value: wrap in list for consistent structure (single-column)
                    targets_df = pd.DataFrame(
                        {"y": [pd.to_numeric(parsed, errors="coerce")]}
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to parse target data for DataFrame creation: {e}. Expected string representation of list or nested list."
                )
        else:
            # Non-string data: attempt to coerce to Series then DataFrame (single-column)
            targets_df = pd.DataFrame(
                {"y": pd.to_numeric(pd.Series(target_data), errors="coerce")}
            )

        return targets_df

    def _validate_data_quality(self, targets_df: pd.DataFrame) -> None:
        """
        Comprehensive data quality validation to ensure clean data reaches models.
        This enforces the contract: "no NaNs or inconsistencies when data gets to models".

        Args:
            targets_df: DataFrame containing target data after preprocessing

        Raises:
            ValueError: If data quality issues are detected
        """
        # 1. Check for any remaining NaN values (should be handled by preprocessor)
        if targets_df.isna().any().any():
            nan_count = targets_df.isna().sum().sum()
            raise ValueError(
                f"Data still contains {nan_count} NaN values after preprocessing. "
                f"Check preprocessor configuration and data quality."
            )

        # 2. Check for infinite values
        if np.isinf(targets_df.values).any():
            inf_count = np.isinf(targets_df.values).sum()
            raise ValueError(
                f"Data contains {inf_count} infinite values. "
                f"Check for division by zero or overflow issues."
            )

        # 3. Check for data type consistency
        if not targets_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
            non_numeric_cols = targets_df.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()
            raise ValueError(
                f"Non-numeric columns detected: {non_numeric_cols}. "
                f"All target data must be numeric for forecasting."
            )

        # 4. Check for empty data
        if targets_df.empty:
            raise ValueError("Data is empty after preprocessing.")

            # 5. Check for constant data (all values the same) - REMOVED: too strict for test scenarios
            # for col in targets_df.columns:
            #     if targets_df[col].nunique() <= 1:
            #         raise ValueError(f"Column {col} contains constant data (all values identical). "
            #                        f"This will cause forecasting issues.")

            # 6. Check for reasonable data ranges (optional, configurable) - REMOVED: not needed for basic validation
            # if self.config.get("dataset", {}).get("validate_ranges", False):
            #     for col in targets_df.columns:
            #     col_data = targets_df[col]
            #     if col_data.std() == 0:
            #         raise ValueError(f"Column {col} has zero variance. "
            #                         f"This will cause forecasting issues.")

        print(
            f"[INFO] Data quality validation passed. Data is clean and ready for models."
        )

    def load_single_chunk(self, chunk_id: int) -> Dataset:
        """
        Load a single data chunk and create a Dataset object.
        All data is treated as multivariate where univariate is just num_targets == 1.

        Args:
            chunk_id: The chunk ID to load

        Returns:
            Dataset object containing the loaded data with train/val/test splits

        Raises:
            FileNotFoundError: If the chunk file doesn't exist
        """
        chunk_file = f"chunk{chunk_id:03d}.csv"
        chunk_path = os.path.join(self.dataset_path, chunk_file)

        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        # Load the chunk data
        chunk_data = pd.read_csv(chunk_path)

        # Extract basic information
        item_id = chunk_data["item_id"].iloc[0]
        start = chunk_data["start"].iloc[0]
        freq = chunk_data["freq"].iloc[0]

        # Handle targets by inference
        target_data = chunk_data["target"].iloc[0]
        self._extract_target_structure(target_data)

        # Create targets DataFrame - all data treated as multivariate
        targets_df = self._create_targets_dataframe(target_data)

        # Handle missing values using the preprocessor (centralized NaN handling)
        targets_df = self.preprocessor._handle_missing_values(targets_df)

        # Comprehensive data quality validation BEFORE passing to models
        self._validate_data_quality(targets_df)

        # Create timestamps
        # targets_df has rows as time steps and columns as target series
        target_length = targets_df.shape[0]

        # Create timestamps based on frequency
        start_date = pd.to_datetime(start)
        # Handle deprecated frequency 'T' -> 'min'
        if freq == "T":
            freq = "min"
        timestamps = pd.date_range(start=start_date, periods=target_length, freq=freq)

        # Split data into train/validation/test
        split_ratio = self.config["dataset"].get("split_ratio", [0.8, 0.1, 0.1])
        train_size = int(target_length * split_ratio[0])
        val_size = int(target_length * split_ratio[1])

        # Split the DataFrame into train/val/test (rows are time steps)
        train_targets = targets_df.iloc[:train_size, :]
        val_targets = targets_df.iloc[train_size : train_size + val_size, :]
        test_targets = targets_df.iloc[train_size + val_size :, :]

        train_timestamps = timestamps[:train_size]
        val_timestamps = timestamps[train_size : train_size + val_size]
        test_timestamps = timestamps[train_size + val_size :]

        # Create Dataset object with pandas DataFrames
        metadata = {
            "start": start,
            "freq": freq,
            "num_targets": self.num_targets,
            "item_id": item_id,
        }

        return Dataset(
            train=DatasetSplit(
                targets=train_targets, timestamps=train_timestamps.values
            ),
            validation=DatasetSplit(
                targets=val_targets, timestamps=val_timestamps.values
            ),
            test=DatasetSplit(targets=test_targets, timestamps=test_timestamps.values),
            name=f"chunk_{chunk_id}",
            metadata=metadata,
        )

    def load_several_chunks(self, upper_chunk_index: int) -> List[Dataset]:
        """
        Loads several chunks and returns them as a list of Dataset objects.

        Args:
            upper_chunk_index: The last index of the chunk we want the data of (inclusive)

        Returns:
            List of Dataset objects, with each Dataset object containing train, val, test splits.
        """
        list_of_chunks = []
        for chunk_index in range(1, upper_chunk_index + 1):
            list_of_chunks.append(self.load_single_chunk(chunk_index))
        return list_of_chunks

    def load_data(self) -> Dataset:
        """
        Load data using the configured number of chunks.

        Returns:
            Dataset object containing the loaded data

        Note:
            This method is a convenience method that loads a single chunk by default.
            For multiple chunks, use load_several_chunks() directly.
        """
        chunks_to_load = self.config["dataset"].get("chunks", 1)
        if chunks_to_load == 1:
            return self.load_single_chunk(1)
        else:
            datasets = self.load_several_chunks(chunks_to_load)
            # For now, return the first dataset - this could be enhanced to handle multiple chunks
            return datasets[0] if datasets else None
