"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
import random
import re
import ast
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary. Preprocessing-related keys are under 'task'.
                Supports: normalize, handle_missing, max_num_variates.
        """
        self.config = config
        self.normalize = config.get('task', {}).get('dataset', {}).get('normalize', False)
        self.handle_missing = config.get('task', {}).get('dataset', {}).get('handle_missing', 'interpolate')
        self.max_num_variates = config.get('task', {}).get('max_num_variates', None)

    def _parse_and_clean_target(self, target_raw: str) -> np.ndarray:
        """
        Parse and clean raw target string data, handling empty values and converting to numpy array.

        Args:
            target_raw: Raw target data as string (JSON-like array format)

        Returns:
            Cleaned numpy array in (num_time_series, num_time_steps) format
        """
        # Clean the target string to handle empty values before parsing
        print(f"[DEBUG] Raw target string length: {len(target_raw)}")
        print(f"[DEBUG] Raw target preview: {target_raw[:200]}...")
        
        # Replace empty values (consecutive commas) with None
        target_cleaned_str = target_raw
        while ', ,' in target_cleaned_str or ',,' in target_cleaned_str:
            target_cleaned_str = re.sub(r',\s*,', ', None,', target_cleaned_str)
        # Handle leading empty values like "[, 1, 2]" -> "[None, 1, 2]"
        target_cleaned_str = re.sub(r'\[\s*,', '[None,', target_cleaned_str)
        # Handle trailing empty values like "[1, 2, ]" -> "[1, 2, None]"
        target_cleaned_str = re.sub(r',\s*\]', ', None]', target_cleaned_str)
        
        print(f"[DEBUG] Cleaned target string length: {len(target_cleaned_str)}")
        
        try:
            target_parsed = ast.literal_eval(target_cleaned_str)
            print(f"[DEBUG] Successfully parsed target with ast.literal_eval")
        except SyntaxError as e:
            print(f"[DEBUG] ast.literal_eval failed: {e}")
            # If still fails, try more aggressive cleaning
            # Replace any remaining empty strings with None
            target_cleaned_str = target_cleaned_str.replace('""', 'None').replace("''", 'None')
            target_parsed = ast.literal_eval(target_cleaned_str)
            print(f"[DEBUG] Successfully parsed after aggressive cleaning")
        
        print(f"[DEBUG] Parsed target type: {type(target_parsed)}")
        if isinstance(target_parsed, list) and len(target_parsed) > 0:
            print(f"[DEBUG] First element type: {type(target_parsed[0])}")
            if isinstance(target_parsed[0], list):
                print(f"[DEBUG] First row length: {len(target_parsed[0])}")
        
        # Detect if univariate (1D list) or multivariate (2D list)
        is_univariate = isinstance(target_parsed, list) and len(target_parsed) > 0 and not isinstance(target_parsed[0], list)
        print(f"[DEBUG] Is univariate: {is_univariate}")
        
        if is_univariate:
            # Convert 1D list to 2D: [v1, v2, ...] -> [[v1, v2, ...]]
            target_parsed = [target_parsed]
            print(f"[DEBUG] Converted to 2D, new length: {len(target_parsed)}")
        
        # Convert None values and empty strings to NaN before creating numpy array
        target_cleaned = []
        for i, row in enumerate(target_parsed):
            cleaned_row = []
            for j, val in enumerate(row):
                if val is None or val == "" or val == "None":
                    cleaned_row.append(np.nan)
                else:
                    try:
                        cleaned_row.append(float(val))
                    except (ValueError, TypeError):
                        cleaned_row.append(np.nan)
            target_cleaned.append(cleaned_row)
            if i < 3:  # Debug first few rows
                print(f"[DEBUG] Row {i} length: {len(cleaned_row)}, first few values: {cleaned_row[:5]}")
        
        target = np.array(target_cleaned)
        print(f"[DEBUG] Final target shape: {target.shape}")
        print(f"[DEBUG] Target dtype: {target.dtype}")
        print(f"[DEBUG] NaN count: {np.isnan(target).sum()}")
        
        return target

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
        # For (num_time_series, num_time_steps) format, time_steps is the second dimension
        data_length = arr.shape[0] if arr.ndim == 1 else arr.shape[1]
        timestamps = pd.date_range(start=start, periods=data_length, freq=freq).values
        timestamps = np.array(timestamps)
        # Convert None values to np.nan and ensure float dtype
        # First convert None to NaN, then convert to float
        if arr.dtype == object:
            # Handle None values in object arrays
            arr = np.where(arr == None, np.nan, arr)
        arr = np.array(arr, dtype=float)
        
        # Ensure we have a 2D array for consistent processing
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr.reshape(1, -1)  # (num_time_series, num_time_steps) for single series
            timestamps = timestamps.reshape(-1, 1)

        result = arr.copy()
        result_timestamps = timestamps.copy()

        if self.handle_missing == 'drop':
            # Remove series with any NaN values (rows in our format)
            valid_series = ~np.isnan(result).any(axis=1)
            result = result[valid_series]
            # Note: timestamps don't change when dropping series
        elif self.handle_missing == 'mean':
            # Fill NaN with series means (row means in our format)
            for series_idx in range(result.shape[0]):
                series_data = result[series_idx, :]
                mean_val = np.nanmean(series_data)
                if not np.isnan(mean_val):
                    series_data[np.isnan(series_data)] = mean_val
                    result[series_idx, :] = series_data
        elif self.handle_missing == 'median':
            # Fill NaN with series medians (row medians in our format)
            for series_idx in range(result.shape[0]):
                series_data = result[series_idx, :]
                median_val = np.nanmedian(series_data)
                if not np.isnan(median_val):
                    series_data[np.isnan(series_data)] = median_val
                    result[series_idx, :] = series_data
        elif self.handle_missing == 'interpolate':
            # Interpolate missing values for each series (row in our format)
            for series_idx in range(result.shape[0]):
                result[series_idx, :] = self._interpolate_column(result[series_idx, :])
        elif self.handle_missing == 'forward_fill':
            # Forward fill missing values for each series
            for series_idx in range(result.shape[0]):
                result[series_idx, :] = self._forward_fill_column(result[series_idx, :])
        elif self.handle_missing == 'backward_fill':
            # Backward fill missing values for each series
            for series_idx in range(result.shape[0]):
                result[series_idx, :] = self._backward_fill_column(result[series_idx, :])
        else:
            # Default: forward fill then backward fill for each series
            for series_idx in range(result.shape[0]):
                series_data = self._forward_fill_column(result[series_idx, :])
                result[series_idx, :] = self._backward_fill_column(series_data)

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

    def _cap_variates(self, target: np.ndarray) -> np.ndarray:
        """
        Cap the number of variates (time series) to max_num_variates if specified.
        
        Args:
            target: Target array in (num_time_series, num_time_steps) format
            
        Returns:
            Target array with capped number of variates
        """
        if self.max_num_variates is None:
            return target
            
        # For 2D arrays, first dimension is num_time_series (variates)
        if target.ndim == 2 and target.shape[0] > self.max_num_variates:
            print(f"[DEBUG] Capping variates from {target.shape[0]} to {self.max_num_variates}")
            target = target[:self.max_num_variates, :]
            print(f"[DEBUG] After capping, target shape: {target.shape}")
        elif target.ndim == 1:
            # For 1D arrays, we only have one variate, so no capping needed
            pass
            
        return target

    def clean(self, time_start: str, freq: str, target_raw: str) -> Tuple[np.ndarray, str, str, np.ndarray]:
        """
        Clean raw target data by parsing, handling missing values, and normalizing.

        Args:
            time_start: Start time as string (will be converted to pandas Timestamp)
            freq: Frequency as string (pandas-compatible frequency)
            target_raw: Raw target data as string (JSON-like array format)

        Returns:
            Tuple of (timestamps, time_start, freq, target) - cleaned timestamps, time_start, freq, and target
        """
        print(f"[DEBUG] Starting preprocessor.clean() with time_start={time_start}, freq={freq}")
        
        # 1. Parse and clean the raw target string
        target = self._parse_and_clean_target(target_raw)
        
        # Handle empty arrays
        if target.size == 0:
            raise ValueError("Target array is empty in Preprocessor.clean().")
        
        print(f"[DEBUG] After parsing, target shape: {target.shape}")

        # 2. Cap the number of variates if specified
        target = self._cap_variates(target)
        print(f"[DEBUG] After variate capping, target shape: {target.shape}")

        # 3. Validate/fix time_start
        try:
            pd.Timestamp(time_start)
        except (ValueError, TypeError):
            time_start = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        # 4. Validate/fix freq
        try:
            pd.date_range(start=time_start, periods=2, freq=freq)
        except (ValueError, TypeError):
            valid_freqs = ['D', 'H', 'W']
            freq = random.choice(valid_freqs)

        # 5. Handle missing values (creates timestamps internally)
        print(f"[DEBUG] Before missing value handling, target shape: {target.shape}")
        timestamps_cleaned, target_cleaned = self._handle_missing_values(target, time_start, freq)
        print(f"[DEBUG] After missing value handling, target shape: {target_cleaned.shape}")
        print(f"[DEBUG] Timestamps shape: {timestamps_cleaned.shape}")

        # 6. Normalize if configured
        if self.normalize:
            print(f"[DEBUG] Normalizing data...")
            # Ensure we have a 2D array for StandardScaler
            is_1d = target_cleaned.ndim == 1
            if is_1d:
                target_cleaned = target_cleaned.reshape(-1, 1)

            # Apply StandardScaler (standardization: mean=0, std=1)
            scaler = StandardScaler()
            target_cleaned = scaler.fit_transform(target_cleaned)
            print(f"[DEBUG] After normalization, target shape: {target_cleaned.shape}")

        print(f"[DEBUG] Final result - timestamps: {timestamps_cleaned.shape}, target: {target_cleaned.shape}")
        return timestamps_cleaned, time_start, freq, target_cleaned