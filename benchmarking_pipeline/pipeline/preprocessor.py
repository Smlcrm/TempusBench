"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary. Preprocessing-related keys are under 'dataset'.
                Supports: normalize, handle_missing.
        """
        self.config = config
        self.normalize = config.get('dataset').get('normalize', False)
        self.handle_missing = config.get('dataset').get('handle_missing', 'interpolate')

    def _handle_missing_values(self, arr: np.ndarray, start: str, freq: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values in the ndarray and return cleaned data with corresponding timestamps.

        Args:
            arr: Input array to clean
            start: Start time as string
            freq: Frequency as string

        Returns:
            Tuple of (cleaned_array, cleaned_timestamps)
        """
        # Create timestamps array
        data_length = arr.shape[0] if arr.ndim == 1 else arr.shape[1]
        timestamps = pd.date_range(start=start, periods=data_length, freq=freq).values
        timestamps = np.array(timestamps)
        # Ensure we have a 2D array for consistent processing
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr.reshape(-1, 1)
            timestamps = timestamps.reshape(-1, 1)

        result = arr.copy()
        result_timestamps = timestamps.copy()

        if self.handle_missing == 'drop':
            # Remove rows with any NaN values
            valid_rows = ~np.isnan(result).any(axis=1)
            result = result[valid_rows]
            result_timestamps = result_timestamps[valid_rows]
        elif self.handle_missing == 'mean':
            # Fill NaN with column means
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                mean_val = np.nanmean(col_data)
                if not np.isnan(mean_val):
                    col_data[np.isnan(col_data)] = mean_val
                    result[:, col_idx] = col_data
        elif self.handle_missing == 'median':
            # Fill NaN with column medians
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                median_val = np.nanmedian(col_data)
                if not np.isnan(median_val):
                    col_data[np.isnan(col_data)] = median_val
                    result[:, col_idx] = col_data
        elif self.handle_missing == 'interpolate':
            # Interpolate missing values for each column
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._interpolate_column(result[:, col_idx])
        elif self.handle_missing == 'forward_fill':
            # Forward fill missing values
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._forward_fill_column(result[:, col_idx])
        elif self.handle_missing == 'backward_fill':
            # Backward fill missing values
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._backward_fill_column(result[:, col_idx])
        else:
            # Default: forward fill then backward fill
            for col_idx in range(result.shape[1]):
                col_data = self._forward_fill_column(result[:, col_idx])
                result[:, col_idx] = self._backward_fill_column(col_data)

        return result_timestamps, result

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

    def clean(self, time_start: str, freq: str, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean a single target array by validating time_start and freq, handling missing values, and normalizing.

        Args:
            time_start: Start time as string (will be converted to pandas Timestamp)
            freq: Frequency as string (pandas-compatible frequency)
            target: Target array (univariate or multivariate ndarray)

        Returns:
            Tuple of (target, timestamps) - cleaned target and corresponding timestamps
        """
        # Handle empty arrays
        if target.size == 0:
            raise ValueError("Target array is empty in Preprocessor.clean().")

        # 1. Validate/fix time_start
        try:
            pd.Timestamp(time_start)
        except (ValueError, TypeError):
            time_start = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        # 2. Validate/fix freq
        try:
            pd.date_range(start=time_start, periods=2, freq=freq)
        except (ValueError, TypeError):
            valid_freqs = ['D', 'H', 'W']
            freq = random.choice(valid_freqs)

        # 3. Handle missing values (creates timestamps internally)
        timestamps_cleaned, target_cleaned = self._handle_missing_values(target, time_start, freq)

        # 4. Normalize if configured
        if self.normalize:
            # Ensure we have a 2D array for StandardScaler
            is_1d = target_cleaned.ndim == 1
            if is_1d:
                target_cleaned = target_cleaned.reshape(-1, 1)

            # Apply StandardScaler (standardization: mean=0, std=1)
            scaler = StandardScaler()
            target_cleaned = scaler.fit_transform(target_cleaned)

        return timestamps_cleaned, time_start, freq, target_cleaned