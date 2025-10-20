import os
import sys
import numpy as np
import pytest

# Ensure local workspace package is imported (not a globally installed version)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmarking_pipeline.pipeline.preprocessor import Preprocessor


def _build_config(normalize: bool):
    return {
        "task": {
            "dataset": {
                "normalize": normalize,
                "handle_missing": "interpolate",
            },
            # Keep default: max_num_variates = None
        }
    }


def test_preprocessor_normalizes_univariate_when_enabled():
    config = _build_config(normalize=True)
    pre = Preprocessor(config)

    # Simple univariate series
    target_raw = str([1.0, 2.0, 3.0, 4.0, 5.0])
    time_start = "2020-01-01"
    freq = "D"

    timestamps, ts_start, out_freq, target = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw)

    # Shape: (num_steps, 1)
    assert target.shape == (5, 1)

    # Check approximately zero mean and unit variance per feature
    col = target[:, 0]
    mean = float(np.mean(col))
    std = float(np.std(col, ddof=0))

    assert abs(mean) < 1e-7
    assert abs(std - 1.0) < 1e-6


def test_preprocessor_normalizes_multivariate_when_enabled():
    config = _build_config(normalize=True)
    pre = Preprocessor(config)

    # Two features provided as list of lists (features first), parser will transpose
    # Feature A: 1..5, Feature B: 10..50 step 10
    raw_features = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
    target_raw = str(raw_features)
    time_start = "2020-01-01"
    freq = "D"

    _, _, _, target = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw)

    # Shape: (num_steps, num_features)
    assert target.shape == (5, 2)

    # Check each feature standardized independently
    for j in range(target.shape[1]):
        col = target[:, j]
        mean = float(np.mean(col))
        std = float(np.std(col, ddof=0))
        assert abs(mean) < 1e-7
        assert abs(std - 1.0) < 1e-6


def test_preprocessor_no_normalization_when_disabled():
    config = _build_config(normalize=False)
    pre = Preprocessor(config)

    # Choose values that do not have zero mean or unit variance
    target_raw = str([2.0, 4.0, 6.0, 8.0])
    time_start = "2020-01-01"
    freq = "D"

    _, _, _, target = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw)

    # Shape: (num_steps, 1)
    assert target.shape == (4, 1)

    col = target[:, 0]
    mean = float(np.mean(col))
    std = float(np.std(col, ddof=0))

    # Verify not standardized
    assert abs(mean) > 1e-3
    assert abs(std - 1.0) > 1e-3


