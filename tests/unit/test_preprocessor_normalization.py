import os
import sys
import numpy as np
import pytest

# Ensure local workspace package is imported (not a globally installed version)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from tempus_bench.pipeline.preprocessor import Preprocessor


def test_preprocessor_normalizes_univariate_when_enabled():
    # Create a mock config file for testing
    import tempfile
    import yaml
    
    config_data = {
        "task_path": "*",
        "evaluation": {
            "tuning_loss": "mae",
            "max_windows": 20,
            "max_num_variates": 10,
            "num_samples": 100,
            "num_quantiles": 10,
            "point_forecast_statistic": "mean"
        },
        "model": {
            "exponential_smoothing": {}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        with tempfile.TemporaryDirectory() as logs_dir:
            pre = Preprocessor(config_path, logs_dir)

            # Simple univariate series
            target_raw = str([1.0, 2.0, 3.0, 4.0, 5.0])
            time_start = "2020-01-01"
            freq = "D"
            normalize = True
            handle_missing = "interpolate"

            timestamps, ts_start, out_freq, target, scaler = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw, normalize=normalize, handle_missing=handle_missing)

            # Shape: (num_steps, 1)
            assert target.shape == (5, 1)

            # Check approximately zero mean and unit variance per feature
            col = target[:, 0]
            mean = float(np.mean(col))
            std = float(np.std(col, ddof=0))

            assert abs(mean) < 1e-7
            assert abs(std - 1.0) < 1e-6
    finally:
        os.unlink(config_path)


def test_preprocessor_normalizes_multivariate_when_enabled():
    # Create a mock config file for testing
    import tempfile
    import yaml
    
    config_data = {
        "task_path": "*",
        "evaluation": {
            "tuning_loss": "mae",
            "max_windows": 20,
            "max_num_variates": 10,
            "num_samples": 100,
            "num_quantiles": 10,
            "point_forecast_statistic": "mean"
        },
        "model": {
            "exponential_smoothing": {}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        with tempfile.TemporaryDirectory() as logs_dir:
            pre = Preprocessor(config_path, logs_dir)

            # Two features provided as list of lists (features first), parser will transpose
            # Feature A: 1..5, Feature B: 10..50 step 10
            raw_features = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
            target_raw = str(raw_features)
            time_start = "2020-01-01"
            freq = "D"
            normalize = True
            handle_missing = "interpolate"

            _, _, _, target, scaler = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw, normalize=normalize, handle_missing=handle_missing)

            # Shape: (num_steps, num_targets)
            assert target.shape == (5, 2)

            # Check each feature standardized independently
            for j in range(target.shape[1]):
                col = target[:, j]
                mean = float(np.mean(col))
                std = float(np.std(col, ddof=0))
                assert abs(mean) < 1e-7
                assert abs(std - 1.0) < 1e-6
    finally:
        os.unlink(config_path)


def test_preprocessor_no_normalization_when_disabled():
    # Create a mock config file for testing
    import tempfile
    import yaml
    
    config_data = {
        "task_path": "*",
        "evaluation": {
            "tuning_loss": "mae",
            "max_windows": 20,
            "max_num_variates": 10,
            "num_samples": 100,
            "num_quantiles": 10,
            "point_forecast_statistic": "mean"
        },
        "model": {
            "exponential_smoothing": {}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        with tempfile.TemporaryDirectory() as logs_dir:
            pre = Preprocessor(config_path, logs_dir)

            # Choose values that do not have zero mean or unit variance
            target_raw = str([2.0, 4.0, 6.0, 8.0])
            time_start = "2020-01-01"
            freq = "D"
            normalize = False
            handle_missing = "interpolate"

            _, _, _, target, scaler = pre.clean(time_start=time_start, freq=freq, target_raw=target_raw, normalize=normalize, handle_missing=handle_missing)

            # Shape: (num_steps, 1)
            assert target.shape == (4, 1)

            col = target[:, 0]
            mean = float(np.mean(col))
            std = float(np.std(col, ddof=0))

            # Verify not standardized
            assert abs(mean) > 1e-3
            assert abs(std - 1.0) > 1e-3
    finally:
        os.unlink(config_path)


