"""
Core data types used throughout the benchmarking pipeline.
"""
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
class PreprocessedData:
    """Container for preprocessed data."""
    data: Dataset
    preprocessing_info: Dict[str, Any]

@dataclass
class ModelArtifacts:
    """Container for model artifacts and metadata."""
    model: Any  # The actual model object
    parameters: Dict[str, Any]
    training_history: Optional[Dict[str, List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None 