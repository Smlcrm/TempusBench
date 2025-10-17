"""
Data preprocessing utilities.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import random
from .data_types import Dataset, DatasetSplit, PreprocessedData


class MinMaxScalerSafe:
    def __init__(self, feature_range=(1e-4, 1.0)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X_scaled):
        return self.scaler.inverse_transform(X_scaled)

class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary. Preprocessing-related keys are under 'dataset'.
                Supports: normalize, normalization_method, handle_missing.
        """
        self.config = config
        self.scalers = {}
        self.preprocessing_config = self._extract_preprocessing_config()

    def _extract_preprocessing_config(self) -> Dict[str, Any]:
        """
        Extract preprocessing options from the dataset section.
        """
        dataset_cfg = self.config.get("dataset", {})
        return {
            "normalize": dataset_cfg.get("normalize", True),
            "normalization_method": dataset_cfg.get("normalization_method", "minmax"),
            "handle_missing": dataset_cfg.get("handle_missing", "interpolate")
        }

    def _handle_missing_values(self, arr: np.ndarray) -> np.ndarray:
        """
        Handle missing values in the ndarray.
        """
        strategy = self.preprocessing_config.get('handle_missing', 'interpolate')

        if arr.size == 0:
            return arr

        # Ensure we have a 2D array for consistent processing
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr.reshape(-1, 1)

        result = arr.copy()

        if strategy == 'drop':
            # Remove rows with any NaN values
            valid_rows = ~np.isnan(result).any(axis=1)
            result = result[valid_rows]
        elif strategy == 'mean':
            # Fill NaN with column means
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                mean_val = np.nanmean(col_data)
                if not np.isnan(mean_val):
                    col_data[np.isnan(col_data)] = mean_val
                    result[:, col_idx] = col_data
        elif strategy == 'median':
            # Fill NaN with column medians
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                median_val = np.nanmedian(col_data)
                if not np.isnan(median_val):
                    col_data[np.isnan(col_data)] = median_val
                    result[:, col_idx] = col_data
        elif strategy == 'interpolate':
            # Interpolate missing values for each column
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._interpolate_column(result[:, col_idx])
        elif strategy == 'forward_fill':
            # Forward fill missing values
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._forward_fill_column(result[:, col_idx])
        elif strategy == 'backward_fill':
            # Backward fill missing values
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._backward_fill_column(result[:, col_idx])
        else:
            # Default: forward fill then backward fill
            for col_idx in range(result.shape[1]):
                col_data = self._forward_fill_column(result[:, col_idx])
                result[:, col_idx] = self._backward_fill_column(col_data)

        # Return to original shape if input was 1D
        if is_1d:
            result = result.flatten()
        return result

    def _interpolate_column(self, col_data: np.ndarray) -> np.ndarray:
        """Interpolate missing values in a single column."""
        if not np.isnan(col_data).any():
            return col_data

        # Get valid indices and values
        valid_mask = ~np.isnan(col_data)
        if not valid_mask.any():
            # All values are NaN, fill with 0
            return np.zeros_like(col_data)

        valid_indices = np.where(valid_mask)[0]
        valid_values = col_data[valid_mask]

        if len(valid_indices) == 1:
            # Only one valid value, fill all with that value
            return np.full_like(col_data, valid_values[0])

        # Create interpolation indices for all positions
        all_indices = np.arange(len(col_data))

        # Interpolate using numpy
        interpolated = np.interp(all_indices, valid_indices, valid_values)

        return interpolated

    def _forward_fill_column(self, col_data: np.ndarray) -> np.ndarray:
        """Forward fill missing values in a single column."""
        result = col_data.copy()
        last_valid = None

        for i in range(len(result)):
            if not np.isnan(result[i]):
                last_valid = result[i]
            elif last_valid is not None:
                result[i] = last_valid
        return result

    def _backward_fill_column(self, col_data: np.ndarray) -> np.ndarray:
        """Backward fill missing values in a single column."""
        result = col_data.copy()
        next_valid = None

        for i in range(len(result) - 1, -1, -1):
            if not np.isnan(result[i]):
                next_valid = result[i]
            elif next_valid is not None:
                result[i] = next_valid
        return result

    def _normalize_features(self, arr: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Normalize numerical features using standard or min-max scaling.
        """
        if not self.preprocessing_config.get('normalize', True):
            return arr

        # Handle empty arrays
        if arr.size == 0:
            return arr

        method = self.preprocessing_config.get('normalization_method', 'standard')
        arr_normalized = arr.copy()

        # Ensure we have a 2D array for consistent processing
        is_1d = arr.ndim == 1
        if is_1d:
            arr_normalized = arr_normalized.reshape(-1, 1)

        # Process each column
        for col_idx in range(arr_normalized.shape[1]):
            col_data = arr_normalized[:, col_idx]

            # Skip columns with no valid data
            if np.isnan(col_data).all() or len(col_data[~np.isnan(col_data)]) == 0:
                continue

            # Handle columns with some NaN values by filling them before scaling
            col_mean = np.nanmean(col_data)
            if not np.isnan(col_mean):
                col_data = np.where(np.isnan(col_data), col_mean, col_data)
            else:
                # All values are NaN, skip this column
                continue

            if is_training:
                if method == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScalerSafe()
                arr_normalized[:, col_idx] = scaler.fit_transform(col_data.reshape(-1, 1)).flatten()
                self.scalers[col_idx] = scaler
            else:
                scaler = self.scalers.get(col_idx)
                if scaler:
                    arr_normalized[:, col_idx] = scaler.transform(col_data.reshape(-1, 1)).flatten()

        # Return to original shape if input was 1D
        if is_1d:
            arr_normalized = arr_normalized.flatten()

        return arr_normalized


    def _generate_artificial_timestamps(self, split: DatasetSplit, metadata: Dict[str, Any]) -> DatasetSplit:
        """Generate artificial timestamps for unknown/irregular frequency data."""
        freq = str(metadata.get("freq", "unknown")).lower()

        # Check if we need artificial timestamps
        if freq not in ["unknown", "irregular", "none", ""]:
            return split

        # Get data length
        data_length = split.targets.shape[0]

        # Random frequency >= 1 minute
        frequencies = ["1min", "5min", "15min", "1h", "3h", "1D", "1W"]
        freq = random.choice(frequencies)

        # Generate timestamps from current time
        # Since we removed pandas, we'll create a simple numeric timestamp array
        # representing seconds since epoch
        start_time = datetime.now().timestamp()
        if freq == "1min":
            step = 60
        elif freq == "5min":
            step = 300
        elif freq == "15min":
            step = 900
        elif freq == "1h":
            step = 3600
        elif freq == "3h":
            step = 10800
        elif freq == "1D":
            step = 86400
        elif freq == "1W":
            step = 604800
        else:
            step = 3600  # Default to 1 hour

        timestamps = np.array([start_time + i * step for i in range(data_length)])

        return DatasetSplit(
            targets=split.targets,
            features=split.features,
            metadata=split.metadata,
            timestamps=timestamps
        )

    def preprocess(self, data: Dataset) -> PreprocessedData:
        """
        Apply missing value handling, normalization, and timestamp generation to all splits.

        Args:
            data: Dataset with train/validation/test splits

        Returns:
            PreprocessedData: Transformed dataset and preprocessing metadata
        """
        def process_split(split: DatasetSplit, is_training: bool) -> DatasetSplit:
            # Generate artificial timestamps if needed
            processed_split = self._generate_artificial_timestamps(split, data.metadata or {})

            # Process targets - ensure it's an ndarray
            targets_arr = np.array(processed_split.targets) if not isinstance(processed_split.targets, np.ndarray) else processed_split.targets.copy()
            targets_arr = self._handle_missing_values(targets_arr)
            targets_arr = self._normalize_features(targets_arr, is_training=is_training)

            # Process features (exogenous variables) if present
            features_arr = None
            if processed_split.features is not None:
                features_arr = np.array(processed_split.features) if not isinstance(processed_split.features, np.ndarray) else processed_split.features.copy()
                features_arr = self._handle_missing_values(features_arr)
                features_arr = self._normalize_features(features_arr, is_training=is_training)

            return DatasetSplit(
                targets=targets_arr,
                features=features_arr,
                metadata=processed_split.metadata,
                timestamps=processed_split.timestamps
            )

        train_split = process_split(data.train, is_training=True)
        val_split = process_split(data.validation, is_training=False)
        test_split = process_split(data.test, is_training=False)

        preprocessed_dataset = Dataset(
            train=train_split,
            validation=val_split,
            test=test_split,
            name=data.name,
            metadata=data.metadata
        )

        preprocessing_info = {
            'normalize': self.preprocessing_config.get('normalize', True),
            'normalization_method': self.preprocessing_config.get('normalization_method', 'standard'),
            'handle_missing': self.preprocessing_config.get('handle_missing', 'interpolate'),
            'scalers': {
                col_idx: {
                    'type': type(scaler).__name__,
                    'mean': getattr(scaler, 'mean_', None),
                    'scale': getattr(scaler, 'scale_', None)
                } for col_idx, scaler in self.scalers.items()
            },
            'target_shape': train_split.targets.shape,
            'feature_shape': train_split.features.shape if train_split.features is not None else None
        }

        return PreprocessedData(data=preprocessed_dataset, preprocessing_info=preprocessing_info)

    def clean(self, arr: np.ndarray) -> np.ndarray:
        """
        Clean a single array by handling missing values and normalization.
        This method is used for individual target arrays in the data loader.
        
        Args:
            arr: Input array to clean
            
        Returns:
            Cleaned array with missing values handled and optionally normalized
        """
        if arr.size == 0:
            return arr
            
        # Handle missing values
        cleaned_arr = self._handle_missing_values(arr)
        
        # Normalize if configured
        if self.preprocessing_config.get('normalize', True):
            cleaned_arr = self._normalize_features(cleaned_arr, is_training=True)
            
        return cleaned_arr
