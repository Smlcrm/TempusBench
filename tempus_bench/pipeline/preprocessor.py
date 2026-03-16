"""
Data preprocessing utilities for time series data.

This module provides the Preprocessor class for cleaning raw CSV payloads,
handling missing values, normalizing data, and preparing time series data
for model training and evaluation.
"""

import ast
import re

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.configs import EvaluationConfig, TaskConfig
from ..utils.log_manager import LogManager


class Preprocessor:
    """
    Cleans raw CSV payloads and applies normalization strategies.

    The Preprocessor handles parsing raw target data from CSV files, handling
    missing values using various strategies, normalizing data if configured, and
    ensuring data is in the correct format for model consumption.

    Attributes:
        job_config (JobConfig): Job configuration containing task and evaluation settings.
        evaluation_config (EvaluationConfig): Evaluation-specific configuration.
        logger (LoggerManager): Logger instance for logging operations.
        max_num_variates (Optional[int]): Maximum number of variates to extract from
            dataset. None means all variates.
    """

    def __init__(self, task_config: TaskConfig, evaluation_config: EvaluationConfig):
        """
        Initialize the preprocessor for a concrete task configuration.

        Args:
            task_config: Task configuration object that includes dataset metadata
                and preprocessing directives for the active task.
            evaluation_config: Evaluation configuration object that includes
                benchmark settings for evaluation.
        """
        self.task_config = task_config
        self.evaluation_config = evaluation_config
        self.max_num_variates = self.evaluation_config.max_num_variates

    def _parse_and_clean_target(self, target_raw: str) -> np.ndarray:
        """
        Parse and clean raw target string data, handling empty values.

        This method parses a JSON-like array string from CSV data, handles empty
        values by converting them to NaN, and converts the result to a numpy array
        in the format (num_steps, num_targets).

        Args:
            target_raw (str): Raw target data as string (JSON-like array format).

        Returns:
            np.ndarray: Cleaned numpy array with shape (num_steps, num_targets).
                Empty values are converted to NaN.
        """
        # Clean the target string to handle empty values before parsing
        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Raw target string length: {len(target_raw)}",
        )
        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Raw target preview: {target_raw[:200]}...",
        )

        # Replace empty values (consecutive commas) with None
        target_cleaned_str = target_raw
        while ", ," in target_cleaned_str or ",," in target_cleaned_str:
            target_cleaned_str = re.sub(r",\s*,", ", None,", target_cleaned_str)
        # Handle leading empty values like "[, 1, 2]" -> "[None, 1, 2]"
        target_cleaned_str = re.sub(r"\[\s*,", "[None,", target_cleaned_str)
        # Handle trailing commas by removing them (e.g., "[1, 2, 3, ]" -> "[1, 2, 3]")
        # This ignores trailing commas instead of treating them as empty values
        target_cleaned_str = re.sub(r",\s*\]", "]", target_cleaned_str)

        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Cleaned target string length: {len(target_cleaned_str)}",
        )

        try:
            target_parsed = ast.literal_eval(target_cleaned_str)
            LogManager.get_logger().debug(
                "Preprocessor._parse_and_clean_target",
                f"[DEBUG] Successfully parsed target with ast.literal_eval",
            )
        except SyntaxError as e:
            LogManager.get_logger().debug(
                "Preprocessor._parse_and_clean_target",
                f"[DEBUG] ast.literal_eval failed: {e}",
            )
            # If still fails, try more aggressive cleaning
            target_cleaned_str = target_cleaned_str.replace('""', "None").replace(
                "''", "None"
            )
            target_parsed = ast.literal_eval(target_cleaned_str)
            LogManager.get_logger().debug(
                "Preprocessor._parse_and_clean_target",
                f"[DEBUG] Successfully parsed after aggressive cleaning",
            )

        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Parsed target type: {type(target_parsed)}",
        )

        if isinstance(target_parsed, list) and len(target_parsed) > 0:
            LogManager.get_logger().debug(
                "Preprocessor._parse_and_clean_target",
                f"[DEBUG] First element type: {type(target_parsed[0])}",
            )
            if isinstance(target_parsed[0], list):
                LogManager.get_logger().debug(
                    "Preprocessor._parse_and_clean_target",
                    f"[DEBUG] First row length: {len(target_parsed[0])}",
                )

        # Detect if parsed as univariate: [v1, v2, ...] (1D list)
        is_univariate = (
            isinstance(target_parsed, list)
            and len(target_parsed) > 0
            and not isinstance(target_parsed[0], list)
        )
        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Is univariate: {is_univariate}",
        )

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
                    LogManager.get_logger().debug(
                        "Preprocessor._parse_and_clean_target",
                        f"[DEBUG] Row {i} first value: {target_cleaned[-1][0]}",
                    )
            target = np.array(target_cleaned)  # shape (num_steps, 1)
        else:
            # It is a list of lists: [[feature1], [feature2], ...] -> shape (num_targets, num_steps)
            # We'll treat each sublist as one feature, and transpose after cleaning
            cleaned_features = []
            sequence_lengths = []
            for fi, feature_seq in enumerate(target_parsed):
                if not isinstance(feature_seq, list):
                    # If a feature is not a list, convert it to a list
                    feature_seq = [feature_seq]
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
                        LogManager.get_logger().debug(
                            "Preprocessor._parse_and_clean_target",
                            f"[DEBUG] Feature {fi}, time {ti}, value: {cleaned_seq[-1]}",
                        )
                cleaned_features.append(cleaned_seq)
                sequence_lengths.append(len(cleaned_seq))

            # Check for inconsistent sequence lengths
            if len(set(sequence_lengths)) > 1:
                max_length = max(sequence_lengths)
                LogManager.get_logger().debug(
                    "Preprocessor._parse_and_clean_target",
                    f"[DEBUG] Inconsistent sequence lengths detected: {sequence_lengths}. "
                    f"Padding all sequences to length {max_length} with NaN.",
                )
                # Pad all sequences to the same length with NaN
                for i, seq in enumerate(cleaned_features):
                    if len(seq) < max_length:
                        cleaned_features[i] = seq + [np.nan] * (max_length - len(seq))

            # Transpose: (num_targets, num_steps) -> (num_steps, num_targets)
            try:
                target = np.array(cleaned_features, dtype=float).T
            except (ValueError, TypeError) as e:
                # If still fails, provide detailed error
                lengths_str = ", ".join(str(len(seq)) for seq in cleaned_features)
                raise ValueError(
                    f"Cannot convert cleaned features to numpy array. "
                    f"Feature sequence lengths: [{lengths_str}]. "
                    f"Original error: {e}"
                )

        # Ensure target is always 2D with shape (num_steps, num_targets)
        if target.ndim == 1:
            # If somehow 1D, reshape to (num_steps, 1)
            target = target.reshape(-1, 1)
        elif target.ndim != 2:
            raise ValueError(
                f"Target array must be 2D (num_steps, num_targets) after parsing, "
                f"got shape: {target.shape}, ndim: {target.ndim}"
            )

        # Ensure we have at least one step and one target
        if target.shape[0] == 0:
            raise ValueError(f"Target array has zero steps after parsing (num_steps=0)")
        if target.shape[1] == 0:
            raise ValueError(
                f"Target array has zero targets after parsing (num_targets=0)"
            )

        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Final target shape: {target.shape} (should be (num_steps, num_targets))",
        )
        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] Target dtype: {target.dtype}",
        )
        LogManager.get_logger().debug(
            "Preprocessor._parse_and_clean_target",
            f"[DEBUG] NaN count: {np.isnan(target).sum()}",
        )

        return target

    def _handle_missing_values(
        self, arr: np.ndarray, start: str, freq: str, handle_missing: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values in the ndarray and return cleaned data with timestamps.

        This method applies the specified missing value handling strategy to the input
        array and generates corresponding timestamps based on the start time and frequency.

        Args:
            arr (np.ndarray): Input array to clean with shape (num_steps, num_targets).
            start (str): Start time as string (pandas-compatible).
            freq (str): Frequency as string (pandas-compatible frequency).
            handle_missing (str): Strategy for handling missing values. Options:
                'drop', 'mean', 'median', 'interpolate', 'forward_fill', 'backward_fill'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - timestamps: Array of shape (num_steps,) with generated timestamps.
                - cleaned_array: Array of shape (num_steps, num_targets) with missing
                  values handled according to the strategy.
        """
        # Input requirement: arr shape (num_steps, num_targets)
        num_steps = arr.shape[0]
        # Generate timestamps for each step (length = num_steps)
        # Map deprecated frequency strings to new ones
        freq_mapping = {
            "M": "ME",  # Monthly -> Month End
            "Q": "QE",  # Quarterly -> Quarter End
            "A": "YE",  # Annual -> Year End
        }
        mapped_freq = freq_mapping.get(freq, freq)

        # Convert start to pandas Timestamp if it's a string or numeric
        try:
            if isinstance(start, str):
                start_ts = pd.Timestamp(start)
            elif isinstance(start, (int, float)):
                # Handle Unix timestamps (seconds, milliseconds, microseconds, nanoseconds)
                if start > 1e12:  # Likely nanoseconds or microseconds
                    start_ts = pd.Timestamp(start, unit="ns")
                elif start > 1e9:  # Likely milliseconds
                    start_ts = pd.Timestamp(start, unit="ms")
                else:
                    start_ts = pd.Timestamp(start, unit="s")
            else:
                start_ts = pd.Timestamp(start)
        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(
                f"Invalid start time format: {start!r}. Cannot convert to pandas Timestamp. "
                f"Original error: {e}"
            )

        # Generate date range with error handling for overflow
        try:
            timestamps = pd.date_range(
                start=start_ts, periods=num_steps, freq=mapped_freq
            ).to_numpy()
        except (ValueError, OverflowError) as e:
            # If overflow occurs, try using end parameter instead of periods
            # Calculate end date from start + (num_steps - 1) periods using pandas offsets
            try:
                from pandas.tseries.frequencies import to_offset

                offset = to_offset(mapped_freq)
                # Compute end date: start + (num_steps - 1) * frequency offset
                end_ts = start_ts + offset * (num_steps - 1)
                timestamps = pd.date_range(
                    start=start_ts, end=end_ts, freq=mapped_freq
                ).to_numpy()
            except Exception as e2:
                raise ValueError(
                    f"Cannot generate date range with start={start_ts}, periods={num_steps}, freq={mapped_freq!r}. "
                    f"Original error: {e}. Fallback error: {e2}. "
                    f"This might be due to very large number of periods causing overflow."
                )

        # Convert to float and handle None
        if arr.dtype == object:
            arr = np.where(arr == None, np.nan, arr)
        data = np.array(arr, dtype=float)  # ensure float dtype and copy

        result = data.copy()

        if handle_missing == "drop":
            # Remove rows (timesteps) with any NaN values
            valid_rows = ~np.isnan(result).any(axis=1)
            result = result[valid_rows]
            timestamps = timestamps[valid_rows]
        elif handle_missing == "mean":
            # Fill NaN with column means (feature means)
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                mean_val = np.nanmean(col_data)
                if not np.isnan(mean_val):
                    col_data[np.isnan(col_data)] = mean_val
                    result[:, col_idx] = col_data
        elif handle_missing == "median":
            # Fill NaN with column medians (feature medians)
            for col_idx in range(result.shape[1]):
                col_data = result[:, col_idx]
                median_val = np.nanmedian(col_data)
                if not np.isnan(median_val):
                    col_data[np.isnan(col_data)] = median_val
                    result[:, col_idx] = col_data
        elif handle_missing == "interpolate":
            # Interpolate missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._interpolate_column(result[:, col_idx])
        elif handle_missing == "forward_fill":
            # Forward fill missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._forward_fill_column(result[:, col_idx])
        elif handle_missing == "backward_fill":
            # Backward fill missing values for each column (feature)
            for col_idx in range(result.shape[1]):
                result[:, col_idx] = self._backward_fill_column(result[:, col_idx])
        else:
            # Default: forward fill then backward fill for each column
            for col_idx in range(result.shape[1]):
                col_filled = self._forward_fill_column(result[:, col_idx])
                result[:, col_idx] = self._backward_fill_column(col_filled)

        # Validate output shape is (num_steps, num_targets)
        if result.ndim != 2:
            raise ValueError(
                f"Result array must be 2D (num_steps, num_targets) after handling missing values, "
                f"got shape: {result.shape}, ndim: {result.ndim}"
            )

        if result.shape[0] == 0:
            raise ValueError(
                "Result array has zero steps after handling missing values (num_steps=0)"
            )

        if result.shape[1] == 0:
            raise ValueError(
                "Result array has zero targets after handling missing values (num_targets=0)"
            )

        # Ensure timestamps length matches result steps
        if len(timestamps) != result.shape[0]:
            raise ValueError(
                f"Timestamps length ({len(timestamps)}) does not match "
                f"result num_steps ({result.shape[0]}) after handling missing values"
            )

        return timestamps, result

    def _interpolate_column(self, col_data: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values in a single column.

        Args:
            col_data (np.ndarray): Column data with possible NaN values.

        Returns:
            np.ndarray: Column data with interpolated missing values. If all values
                are NaN, returns zeros.
        """
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
        """
        Forward fill missing values in a single column.

        Args:
            col_data (np.ndarray): Column data with possible NaN values.

        Returns:
            np.ndarray: Column data with forward-filled missing values (carries
                last valid value forward).
        """
        result = col_data.copy()
        last_valid = None

        for i in range(len(result)):
            if not np.isnan(result[i]):
                last_valid = result[i]
            elif last_valid is not None:
                result[i] = last_valid
        return result

    def _backward_fill_column(self, col_data: np.ndarray) -> np.ndarray:
        """
        Backward fill missing values in a single column.

        Args:
            col_data (np.ndarray): Column data with possible NaN values.

        Returns:
            np.ndarray: Column data with backward-filled missing values (carries
                next valid value backward).
        """
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
            target (np.ndarray): Target array in (num_steps, num_targets) format.

        Returns:
            np.ndarray: Target array with capped number of features (columns),
                keeping only the first max_num_variates columns if capping is applied.
        """
        if self.max_num_variates is None or self.max_num_variates == float("inf"):
            return target

        if target.shape[1] > self.max_num_variates:
            LogManager.get_logger().debug(
                "Preprocessor._cap_variates",
                f"[DEBUG] Capping num_targets from {target.shape[1]} to {self.max_num_variates}",
            )
            target = target[:, : self.max_num_variates]
            LogManager.get_logger().debug(
                "Preprocessor._cap_variates",
                f"[DEBUG] After capping, target shape: {target.shape}",
            )

        # Validate output shape is (num_steps, num_targets)
        if target.ndim != 2:
            raise ValueError(
                f"Target array must be 2D (num_steps, num_targets) after capping variates, "
                f"got shape: {target.shape}, ndim: {target.ndim}"
            )

        if target.shape[0] == 0:
            raise ValueError(
                "Target array has zero steps after capping variates (num_steps=0)"
            )

        if target.shape[1] == 0:
            raise ValueError(
                "Target array has zero targets after capping variates (num_targets=0)"
            )

        return target

    def clean(
        self,
        time_start: str,
        freq: str,
        target_raw: str,
        normalize: bool,
        handle_missing: str,
    ) -> Tuple[np.ndarray, str, str, np.ndarray, Optional[StandardScaler]]:
        """
        Clean raw target data by parsing, handling missing values, and normalizing.

        This method performs the complete preprocessing pipeline: parsing raw target
        data, capping variates if configured, validating timestamps and frequency,
        handling missing values, and optionally normalizing the data.

        Args:
            time_start (str): Start time as string (will be converted to pandas Timestamp).
            freq (str): Frequency as string (pandas-compatible frequency).
            target_raw (str): Raw target data as string (JSON-like array format).
            normalize (bool): Whether to normalize the data using StandardScaler.
            handle_missing (str): Strategy for handling missing values. Options:
                'drop', 'mean', 'median', 'interpolate', 'forward_fill', 'backward_fill'.

        Returns:
            Tuple[np.ndarray, str, str, np.ndarray, Optional[StandardScaler]]: A tuple containing:
                - timestamps (np.ndarray): Cleaned timestamps array of shape (num_steps,).
                - time_start (str): Sanitized start timestamp string.
                - freq (str): Validated frequency string.
                - target (np.ndarray): Processed target array of shape (num_steps, num_targets).
                - scaler (Optional[StandardScaler]): Scaler instance if normalization was applied,
                  None otherwise.

        Raises:
            ValueError: If target array is empty, has incorrect dimensions, or frequency
                is invalid or missing.
        """
        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] Starting preprocessor.clean() with time_start={time_start}, freq={freq}",
        )

        # 1. Parse and clean the raw target string
        target = self._parse_and_clean_target(target_raw)

        # Handle empty arrays
        if target.size == 0:
            raise ValueError("Target array is empty in Preprocessor.clean().")

        LogManager.get_logger().debug(
            "Preprocessor.clean", f"[DEBUG] After parsing, target shape: {target.shape}"
        )

        # Target is already in desired shape (num_steps, num_targets)
        if target.ndim != 2:
            raise ValueError(
                f"Target array must be 2D (num_steps, num_targets), got shape: {target.shape}"
            )

        # 2. Cap the number of variates/features if specified
        target = self._cap_variates(target)
        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] After variate capping, target shape: {target.shape}",
        )

        # 3. Validate/fix time_start
        try:
            pd.Timestamp(time_start)
        except (ValueError, TypeError):
            time_start = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # 4. Validate freq strictly (must come from data, no defaults)
        # Map deprecated frequency strings to new ones
        freq_mapping = {
            "M": "ME",  # Monthly -> Month End
            "Q": "QE",  # Quarterly -> Quarter End
            "A": "YE",  # Annual -> Year End
        }
        mapped_freq = freq_mapping.get(freq)

        try:
            pd.date_range(start=time_start, periods=2, freq=mapped_freq)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid or missing frequency from data: {freq!r}. Original error: {e}"
            )

        # 5. Handle missing values (creates timestamps internally)
        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] Before missing value handling, target shape: {target.shape}",
        )
        timestamps_cleaned, target_cleaned = self._handle_missing_values(
            target, time_start, freq, handle_missing
        )
        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] After missing value handling, target shape: {target_cleaned.shape}",
        )
        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] Timestamps shape: {timestamps_cleaned.shape}",
        )

        # 6. Normalize if configured
        scaler = None
        if normalize:
            LogManager.get_logger().debug(
                "Preprocessor.clean", f"[DEBUG] Normalizing data..."
            )
            # target_cleaned: (num_steps, num_targets)
            scaler = StandardScaler()
            target_cleaned = scaler.fit_transform(target_cleaned)
            LogManager.get_logger().debug(
                "Preprocessor.clean",
                f"[DEBUG] After normalization, target shape: {target_cleaned.shape}",
            )

        # Final validation: ensure target shape is (num_steps, num_targets)
        if target_cleaned.ndim != 2:
            raise ValueError(
                f"Target array must be 2D (num_steps, num_targets) after cleaning, "
                f"got shape: {target_cleaned.shape}, ndim: {target_cleaned.ndim}"
            )

        num_steps_actual = target_cleaned.shape[0]
        num_targets_actual = target_cleaned.shape[1]

        if num_steps_actual == 0:
            raise ValueError("Target array has zero steps after cleaning (num_steps=0)")

        if num_targets_actual == 0:
            raise ValueError(
                "Target array has zero targets after cleaning (num_targets=0)"
            )

        # Ensure timestamps match target steps
        if len(timestamps_cleaned) != num_steps_actual:
            raise ValueError(
                f"Timestamps length ({len(timestamps_cleaned)}) does not match "
                f"target num_steps ({num_steps_actual})"
            )

        LogManager.get_logger().debug(
            "Preprocessor.clean",
            f"[DEBUG] Final result - timestamps: {timestamps_cleaned.shape}, target: {target_cleaned.shape} (num_steps={num_steps_actual}, num_targets={num_targets_actual})",
        )
        return timestamps_cleaned, time_start, freq, target_cleaned, scaler
