"""
Data preprocessing utilities.
"""
import ast
import re

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config.configs import JobConfig

class Preprocessor:
    """Clean raw CSV payloads and apply normalization strategies."""

    def __init__(self, config: JobConfig):
        """
        Initialize the preprocessor for a concrete task configuration.

        Args:
            config: `JobConfig` emitted by the configuration manager; supplies
                the task metadata and preprocessing directives.
        """
        self.job_config = config
        self.config = config.benchmark_config
        self.logger = config.logger
        self.max_num_variates = self.config.evaluation.max_num_variates

    def _parse_and_clean_target(self, target_raw: str) -> np.ndarray:
        """
        Parse and clean raw target string data, handling empty values and converting to numpy array.

        Args:
            target_raw: Raw target data as string (JSON-like array format)

        Returns:
            Cleaned numpy array in (num_steps, num_targets) format
        """
        # Clean the target string to handle empty values before parsing
        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Raw target string length: {len(target_raw)}")
        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Raw target preview: {target_raw[:200]}...")

        # Replace empty values (consecutive commas) with None
        target_cleaned_str = target_raw
        while ', ,' in target_cleaned_str or ',,' in target_cleaned_str:
            target_cleaned_str = re.sub(r',\s*,', ', None,', target_cleaned_str)
        # Handle leading empty values like "[, 1, 2]" -> "[None, 1, 2]"
        target_cleaned_str = re.sub(r'\[\s*,', '[None,', target_cleaned_str)
        # Handle trailing empty values like "[1, 2, ]" -> "[1, 2, None]"
        target_cleaned_str = re.sub(r',\s*\]', ', None]', target_cleaned_str)

        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Cleaned target string length: {len(target_cleaned_str)}")

        try:
            target_parsed = ast.literal_eval(target_cleaned_str)
            self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Successfully parsed target with ast.literal_eval")
        except SyntaxError as e:
            self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] ast.literal_eval failed: {e}")
            # If still fails, try more aggressive cleaning
            target_cleaned_str = target_cleaned_str.replace('""', 'None').replace("''", 'None')
            target_parsed = ast.literal_eval(target_cleaned_str)
            self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Successfully parsed after aggressive cleaning")

        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Parsed target type: {type(target_parsed)}")

        if isinstance(target_parsed, list) and len(target_parsed) > 0:
            self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] First element type: {type(target_parsed[0])}")
            if isinstance(target_parsed[0], list):
                self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] First row length: {len(target_parsed[0])}")

        # Detect if parsed as univariate: [v1, v2, ...] (1D list)
        is_univariate = isinstance(target_parsed, list) and len(target_parsed) > 0 and not isinstance(target_parsed[0], list)
        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Is univariate: {is_univariate}")

        if is_univariate:
            # Convert 1D list to column vector of shape (num_steps, 1)
            target_cleaned = []
            for i, val in enumerate(target_parsed):
                if val is None or val == "" or val == "None":
                    target_cleaned.append([np.nan])
                else:
                    try:
                        target_cleaned.append([float(val)])
                    except (ValueError, TypeError):
                        target_cleaned.append([np.nan])
                if i < 3:
                    self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Row {i} first value: {target_cleaned[-1][0]}")
            target = np.array(target_cleaned)  # shape (num_steps, 1)
        else:
            # It is a list of lists: [[feature1], [feature2], ...] -> shape (num_targets, num_steps)
            # We'll treat each sublist as one feature, and transpose after cleaning
            cleaned_features = []
            for fi, feature_seq in enumerate(target_parsed):
                cleaned_seq = []
                for ti, val in enumerate(feature_seq):
                    if val is None or val == "" or val == "None":
                        cleaned_seq.append(np.nan)
                    else:
                        try:
                            cleaned_seq.append(float(val))
                        except (ValueError, TypeError):
                            cleaned_seq.append(np.nan)
                    if fi < 2 and ti < 3:
                        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Feature {fi}, time {ti}, value: {cleaned_seq[-1]}")
                cleaned_features.append(cleaned_seq)
            # Transpose: (num_targets, num_steps) -> (num_steps, num_targets)
            target = np.array(cleaned_features, dtype=float).T

        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Final target shape: {target.shape} (should be (num_steps, num_targets))")
        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] Target dtype: {target.dtype}")
        self.logger.debug("Preprocessor._parse_and_clean_target", f"[DEBUG] NaN count: {np.isnan(target).sum()}")

        return target

    def _handle_missing_values(self, arr: np.ndarray, start: str, freq: str, handle_missing: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values in the ndarray and return cleaned data with corresponding timestamps.

        Args:
            arr: Input array to clean, shape (num_steps, num_targets)
            start: Start time as string
            freq: Frequency as string
            handle_missing: Strategy for handling missing values

        Returns:
            Tuple of (timestamps, cleaned_array)
                - timestamps: shape (num_steps,)
                - cleaned_array: shape (num_steps, num_targets)
        """
        # Input requirement: arr shape (num_steps, num_targets)
        num_steps = arr.shape[0]
        # Generate timestamps for each step (length = num_steps)
        # Map deprecated frequency strings to new ones
        freq_mapping = {
            'M': 'ME',  # Monthly -> Month End
            'Q': 'QE',  # Quarterly -> Quarter End
            'A': 'YE',  # Annual -> Year End
        }
        mapped_freq = freq_mapping.get(freq)
        timestamps = pd.date_range(start=start, periods=num_steps, freq=mapped_freq).to_numpy()

        # Convert to float and handle None
        if arr.dtype == object:
            arr = np.where(arr == None, np.nan, arr)
        data = np.array(arr, dtype=float)  # ensure float dtype and copy

        result = data.copy()

        if handle_missing == 'drop':
            # Remove rows (timesteps) with any NaN values
            valid_rows = ~np.isnan(result).any(axis=1)
            result = result[valid_rows]
            timestamps = timestamps[valid_rows]
        elif handle_missing == 'mean':
            # Fill NaN with column means (feature means)
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                mean_val = np.nanmean(col_data)
                if not np.isnan(mean_val):
                    col_data[np.isnan(col_data)] = mean_val
                    result[:, col_idx] = col_data
        elif handle_missing == 'median':
            # Fill NaN with column medians (feature medians)
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                median_val = np.nanmedian(col_data)
                if not np.isnan(median_val):
                    col_data[np.isnan(col_data)] = median_val
                    result[:, col_idx] = col_data
        elif handle_missing == 'interpolate':
            # Interpolate missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._interpolate_column(result[:, col_idx])
        elif handle_missing == 'forward_fill':
            # Forward fill missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._forward_fill_column(result[:, col_idx])
        elif handle_missing == 'backward_fill':
            # Backward fill missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._backward_fill_column(result[:, col_idx])
        else:
            # Default: forward fill then backward fill for each column
            for col_idx in range(result.shape[1]):
                col_filled = self._forward_fill_column(result[:, col_idx])
                result[:, col_idx] = self._backward_fill_column(col_filled)

        return timestamps, result

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
        Cap the number of features (variates/columns) to max_num_variates if specified.

        Args:
            target: Target array in (num_steps, num_targets) format

        Returns:
            Target array with capped number of features
        """
        if self.max_num_variates is None or self.max_num_variates == float('inf'):
            return target

        if target.shape[1] > self.max_num_variates:
            self.logger.debug("Preprocessor._cap_variates", f"[DEBUG] Capping num_targets from {target.shape[1]} to {self.max_num_variates}")
            target = target[:, :self.max_num_variates]
            self.logger.debug("Preprocessor._cap_variates", f"[DEBUG] After capping, target shape: {target.shape}")

        return target

    def clean(self, time_start: str, freq: str, target_raw: str, normalize: bool, handle_missing: str) -> Tuple[np.ndarray, str, str, np.ndarray, Optional[StandardScaler]]:
        """
        Clean raw target data by parsing, handling missing values, and normalizing.

        Args:
            time_start: Start time as string (will be converted to pandas Timestamp)
            freq: Frequency as string (pandas-compatible frequency)
            target_raw: Raw target data as string (JSON-like array format)
            normalize: Whether to normalize the data using StandardScaler
            handle_missing: Strategy for handling missing values ('drop', 'mean', 'median', 'interpolate', 'forward_fill', 'backward_fill')

        Returns:
            Tuple[np.ndarray, str, str, np.ndarray, Optional[StandardScaler]]:
                Cleaned timestamps, sanitized start timestamp, validated frequency string,
                processed target array (shape `(num_steps, num_targets)`), and an optional scaler.
        """
        self.logger.debug("Preprocessor.clean", f"[DEBUG] Starting preprocessor.clean() with time_start={time_start}, freq={freq}")

        # 1. Parse and clean the raw target string
        target = self._parse_and_clean_target(target_raw)

        # Handle empty arrays
        if target.size == 0:
            raise ValueError("Target array is empty in Preprocessor.clean().")

        self.logger.debug("Preprocessor.clean", f"[DEBUG] After parsing, target shape: {target.shape}")

        # Target is already in desired shape (num_steps, num_targets)
        if target.ndim != 2:
            raise ValueError(f"Target array must be 2D (num_steps, num_targets), got shape: {target.shape}")

        # 2. Cap the number of variates/features if specified
        target = self._cap_variates(target)
        self.logger.debug("Preprocessor.clean", f"[DEBUG] After variate capping, target shape: {target.shape}")

        # 3. Validate/fix time_start
        try:
            pd.Timestamp(time_start)
        except (ValueError, TypeError):
            time_start = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        # 4. Validate freq strictly (must come from data, no defaults)
        # Map deprecated frequency strings to new ones
        freq_mapping = {
            'M': 'ME',  # Monthly -> Month End
            'Q': 'QE',  # Quarterly -> Quarter End
            'A': 'YE',  # Annual -> Year End
        }
        mapped_freq = freq_mapping.get(freq)

        try:
            pd.date_range(start=time_start, periods=2, freq=mapped_freq)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid or missing frequency from data: {freq!r}. Original error: {e}")

        # 5. Handle missing values (creates timestamps internally)
        self.logger.debug("Preprocessor.clean", f"[DEBUG] Before missing value handling, target shape: {target.shape}")
        timestamps_cleaned, target_cleaned = self._handle_missing_values(target, time_start, freq, handle_missing)
        self.logger.debug("Preprocessor.clean", f"[DEBUG] After missing value handling, target shape: {target_cleaned.shape}")
        self.logger.debug("Preprocessor.clean", f"[DEBUG] Timestamps shape: {timestamps_cleaned.shape}")

        # 6. Normalize if configured
        scaler = None
        if normalize:
            self.logger.debug("Preprocessor.clean", f"[DEBUG] Normalizing data...")
            # target_cleaned: (num_steps, num_targets)
            scaler = StandardScaler()
            target_cleaned = scaler.fit_transform(target_cleaned)
            self.logger.debug("Preprocessor.clean", f"[DEBUG] After normalization, target shape: {target_cleaned.shape}")

        self.logger.debug("Preprocessor.clean", f"[DEBUG] Final result - timestamps: {timestamps_cleaned.shape}, target: {target_cleaned.shape}")
        return timestamps_cleaned, time_start, freq, target_cleaned, scaler
