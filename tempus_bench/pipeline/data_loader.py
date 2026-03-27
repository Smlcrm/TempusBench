"""
Data Loader for loading and preprocessing time series data.

This module provides the DataLoader class for handling time series data in CSV format.
It loads complete dataset files and creates Dataset objects for machine learning workflows.

Key Features:
    - Complete CSV dataset loading (not chunked)
    - Tabular schema: variable_name, timestamps, values, variable_type (targets and covariates)
    - Rolling window generation with configurable splits
    - Integration with preprocessing pipeline

The DataLoader treats all data as multivariate where univariate is simply num_targets == 1.
Targets are kept as raw arrays without artificial column naming for maximum flexibility.

Example:
    >>> loader = DataLoader(task_config, evaluation_config)
    >>> for window_idx, dataset in loader.dataset.generate_dataset_split(
    ...     steps=[('context', 24), ('train', 12), ('validate', 6)],
    ...     stride=1,
    ...     max_windows=5,
    ... ):
    ...     # Process dataset...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, List, Set

import numpy as np
import polars as pl

from ..utils.configs import EvaluationConfig, TaskConfig
from .data_types import Dataset
from .preprocessor import Preprocessor

_DATASET_COLUMNS: Final[Set[str]] = {
    "variable_name",
    "timestamps",
    "values",
    "variable_type",
}


def _infer_freq_from_iso_timestamp_strings(timestamps_str: List[str]) -> str:
    """
    Infer a pandas-compatible frequency string from a calendar index (UTC ISO strings).

    Drives Preprocessor date_range; when row counts match, covariate loaders replace
    synthetic timestamps with the parsed calendar grid.
    """
    if len(timestamps_str) <= 1:
        return "D"
    parsed = pl.Series("t", timestamps_str).str.to_datetime(
        time_unit="ns",
        time_zone="UTC",
    )
    epoch_ns = parsed.dt.epoch(time_unit="ns").to_numpy().astype(np.int64)
    deltas_ns = epoch_ns[1:] - epoch_ns[:-1]
    deltas_sec = deltas_ns[deltas_ns > 0].astype(np.float64) / 1e9
    if deltas_sec.size == 0:
        return "D"
    median_sec = float(np.median(deltas_sec))
    if not np.isfinite(median_sec) or median_sec <= 0:
        return "D"
    if median_sec < 60:
        return "s"
    if median_sec < 3600:
        return "min"
    if median_sec < 86400:
        return "h"
    if 5 * 86400 < median_sec < 9 * 86400:
        return "W"
    if 25 * 86400 < median_sec < 35 * 86400:
        return "ME"
    if 300 * 86400 < median_sec < 420 * 86400:
        return "YE"
    if median_sec < 300 * 86400:
        return "d"
    return "YE"


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

    def __init__(
        self,
        task_config: TaskConfig,
        evaluation_config: EvaluationConfig,
        *,
        force_no_normalize: bool = False,
    ):
        """
        Initialize the loader for a specific task and evaluation configuration.

        Args:
            task_config: Task configuration object that includes dataset metadata
                and preprocessing directives for the active task.
            evaluation_config: Evaluation configuration object that includes
                benchmark settings for evaluation.
            force_no_normalize: If True, skip ``StandardScaler`` on targets (same missing-value
                handling as training). Used to export UI display series in raw units.
        """
        self.task_config = task_config
        self.evaluation_config = evaluation_config
        self._force_no_normalize = force_no_normalize
        task_path = Path(self.task_config.task_path)
        self.dataset_path = task_path / self.task_config.dataset.file_name

        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        Load a complete dataset CSV and build a Dataset via the preprocessor.

        The CSV must have columns: variable_name, timestamps, values, variable_type.
        Each ``timestamps`` cell is a JSON array of ISO 8601 strings (UTC ``Z`` recommended).
        Each ``values`` cell is a JSON array of numbers (or null for missing).

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            ValueError: If the dataset cannot be processed or has invalid format.
        """
        file_data = pl.read_csv(self.dataset_path, encoding="utf8")
        columns = set(file_data.columns)
        if columns != _DATASET_COLUMNS:
            raise ValueError(
                f"Dataset CSV must have exactly columns {sorted(_DATASET_COLUMNS)}; "
                f"got {sorted(columns)} ({self.dataset_path})"
            )

        self._load_tabular_dataset(file_data)

    def _load_tabular_dataset(self, file_data: pl.DataFrame) -> None:
        """
        Load targets and optional covariates from the standard tabular schema.

        Args:
            file_data: Polars DataFrame with columns variable_name, timestamps, values, variable_type
        """
        target_rows = file_data.filter(pl.col("variable_type") == "target")
        covariate_rows = file_data.filter(pl.col("variable_type") == "covariate")

        if target_rows.height == 0:
            raise ValueError("Dataset must have at least one row with variable_type 'target'")

        first_target = target_rows.row(0, named=True)
        timestamps_str = json.loads(first_target["timestamps"])
        if not isinstance(timestamps_str, list) or not timestamps_str:
            raise ValueError("Target timestamps must be a non-empty JSON array")
        if not all(isinstance(x, str) for x in timestamps_str):
            raise ValueError(
                "Timestamps must be ISO 8601 strings only (no numeric epochs or bare years)"
            )

        time_freq = _infer_freq_from_iso_timestamp_strings(timestamps_str)
        time_start = timestamps_str[0]

        ts_series = pl.Series("t", timestamps_str).str.to_datetime(
            time_unit="ns",
            time_zone="UTC",
        )
        ts_calendar = ts_series.to_numpy()

        target_data: List[list] = []
        for row in target_rows.iter_rows(named=True):
            values = json.loads(row["values"])
            target_data.append(values)

        target_raw_str = str(target_data)

        covariate_data: List[list] = []
        for row in covariate_rows.iter_rows(named=True):
            values = json.loads(row["values"])
            covariate_data.append(values)

        normalize = (
            False
            if self._force_no_normalize
            else self.task_config.dataset.normalize
        )
        handle_missing = self.task_config.dataset.handle_missing

        preprocessor = Preprocessor(self.task_config, self.evaluation_config)

        timestamps, time_start, time_freq, target, scaler = preprocessor.clean(
            time_start,
            time_freq,
            target_raw_str,
            normalize,
            handle_missing,
        )

        if len(ts_calendar) == len(timestamps):
            timestamps = ts_calendar
            time_start = timestamps_str[0]

        if covariate_data:
            covariate_raw_str = str(covariate_data)
            _, _, _, covariate, _ = preprocessor.clean(
                time_start,
                time_freq,
                covariate_raw_str,
                normalize=False,
                handle_missing=handle_missing,
            )
        else:
            covariate = None

        self.scaler = scaler

        assert target.ndim == 2  # (num_steps, num_targets)
        assert timestamps.ndim == 1  # (num_steps,)
        if covariate is not None:
            assert covariate.ndim == 2  # (num_steps, num_covariates)
            assert covariate.shape[0] == target.shape[0]

        meta = {
            "dataset_path": str(self.dataset_path),
            "time_start": time_start,
            "time_freq": time_freq,
            "num_targets": target.shape[1],
            "num_covariates": covariate.shape[1] if covariate is not None else 0,
        }
        if scaler is not None:
            meta["target_scaler_mean"] = scaler.mean_.tolist()
            meta["target_scaler_scale"] = scaler.scale_.tolist()

        self.dataset = Dataset(
            timestamps=timestamps.tolist(),
            target=target.tolist(),
            covariate=covariate.tolist() if covariate is not None else None,
            metadata=meta,
        )
