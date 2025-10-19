"""
Core data types used throughout the benchmarking pipeline.
"""
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

@dataclass
class DatasetSplit:
    """
    Represents a dataset split (train/val/test/etc) for time series, as actually produced by DataLoader.
    """
    start: int
    end: int
    metadata: Optional[Dict[str, Any]] = None  # Arbitrary additional per-split metadata

@dataclass
class Dataset:
    """
    Container for all dataset splits.
    """
    timestamps: np.ndarray # Array of timestamps (same length as num_steps) or None
    target: np.ndarray  # 2D np.ndarray of Target values
    context: DatasetSplit
    train: DatasetSplit
    validation: DatasetSplit
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TaskResult:
    """
    Contains results of a model task, including hyperparameter optimization and dataset metadata.
    """
    optimal_hyperparameters: Dict[int, Dict[str, float]]
    final_evaluations: Dict[str, float]
    dataset_path: str
    context_window: int
    forecast_horizon: int
    task_type: Literal["deterministic", "stochastic"]
    tuning_loss: str
    dataset_normalize: bool
    dataset_handle_missing: Literal["interpolate", "mean", "median", "drop", "forward_fill", "backward_fill"]
