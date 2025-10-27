"""
Unit tests for path utilities in tempus_bench.utils.paths.

This test verifies that all path functions generate correct absolute paths
based on the project structure.
"""

import pytest
from pathlib import Path

from tempus_bench.utils.paths import (
    get_project_root,
    get_tasks_dir,
    get_models_dir,
    get_configs_dir,
    get_runs_dir,
    get_logs_path,
    get_task_path,
    get_model_path,
    ensure_directory_exists,
)


class TestPathFunctions:
    """Test suite for path utility functions."""
    
    def test_get_project_root(self):
        """Test that get_project_root returns the correct project root path."""
        # The project root is the directory containing tempus_bench
        # Since we're in tests/unit, go up 3 levels to reach project root
        expected_root = Path(__file__).parent.parent.parent
        actual_root = get_project_root()
        
        print(f"\nProject Root:")
        print(f"  Expected: {expected_root}")
        print(f"  Actual:   {actual_root}")
        
        assert actual_root == expected_root, f"Project root mismatch: expected {expected_root}, got {actual_root}"
        assert actual_root.exists(), "Project root should exist"
        
        # Verify it contains tempus_bench directory
        tempus_bench_dir = actual_root / "tempus_bench"
        assert tempus_bench_dir.exists(), "Should contain tempus_bench directory"
    
    def test_get_tasks_dir(self):
        """Test that get_tasks_dir returns the correct tasks directory path."""
        expected = get_project_root() / "tempus_bench" / "tasks"
        actual = get_tasks_dir()
        
        print(f"\nTasks Directory:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Tasks directory path mismatch"
    
    def test_get_models_dir(self):
        """Test that get_models_dir returns the correct models directory path."""
        expected = get_project_root() / "tempus_bench" / "models"
        actual = get_models_dir()
        
        print(f"\nModels Directory:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Models directory path mismatch"
    
    def test_get_configs_dir(self):
        """Test that get_configs_dir returns the correct configs directory path."""
        expected = get_project_root() / "tempus_bench" / "config"
        actual = get_configs_dir()
        
        print(f"\nConfigs Directory:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Configs directory path mismatch"
    
    def test_get_runs_dir(self):
        """Test that get_runs_dir returns the correct runs directory path."""
        expected = get_project_root() / "runs"
        actual = get_runs_dir()
        
        print(f"\nRuns Directory:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Runs directory path mismatch"
    
    def test_get_logs_path(self):
        """Test that get_logs_path returns the correct logs directory path."""
        expected = get_project_root() / "logs"
        actual = get_logs_path()
        
        print(f"\nLogs Directory:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Logs directory path mismatch"
    
    def test_get_task_path(self):
        """Test that get_task_path returns correct task-specific paths."""
        task_name = "multivariate/baggage_100_multivariate"
        expected = get_tasks_dir() / task_name
        actual = get_task_path(task_name)
        
        print(f"\nTask Path:")
        print(f"  Task: {task_name}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Task path mismatch"
        
        # Test with another task
        task_name2 = "univariate/daily_gpu_price_univariate"
        expected2 = get_tasks_dir() / task_name2
        actual2 = get_task_path(task_name2)
        
        print(f"\nTask Path 2:")
        print(f"  Task: {task_name2}")
        print(f"  Expected: {expected2}")
        print(f"  Actual:   {actual2}")
        
        assert actual2 == expected2, "Task path 2 mismatch"
    
    def test_get_model_path(self):
        """Test that get_model_path returns correct model-specific paths."""
        model_type = "deterministic"
        model_name = "arima"
        expected = get_models_dir() / model_type / model_name
        actual = get_model_path(model_type, model_name)
        
        print(f"\nModel Path:")
        print(f"  Type: {model_type}, Name: {model_name}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        
        assert actual == expected, "Model path mismatch"
        
        # Test with stochastic model
        model_type2 = "stochastic"
        model_name2 = "chronos"
        expected2 = get_models_dir() / model_type2 / model_name2
        actual2 = get_model_path(model_type2, model_name2)
        
        print(f"\nModel Path 2:")
        print(f"  Type: {model_type2}, Name: {model_name2}")
        print(f"  Expected: {expected2}")
        print(f"  Actual:   {actual2}")
        
        assert actual2 == expected2, "Model path 2 mismatch"
    
    def test_path_consistency(self):
        """Test that all base paths are consistent with each other."""
        project_root = get_project_root()
        
        print(f"\nPath Consistency Check:")
        print(f"  Project Root: {project_root}")
        
        # All should be under project_root
        tasks = get_tasks_dir()
        models = get_models_dir()
        configs = get_configs_dir()
        runs = get_runs_dir()
        logs = get_logs_path()
        
        assert tasks.is_relative_to(project_root) or tasks == project_root
        assert models.is_relative_to(project_root) or models == project_root
        assert configs.is_relative_to(project_root) or configs == project_root
        assert runs.is_relative_to(project_root) or runs == project_root
        assert logs.is_relative_to(project_root) or logs == project_root
        
        print(f"  ✓ All paths are consistent with project root")
    
    def test_paths_are_absolute(self):
        """Test that all returned paths are absolute paths."""
        print(f"\nPath Absolute Check:")
        
        paths = [
            ("project_root", get_project_root()),
            ("tasks_dir", get_tasks_dir()),
            ("models_dir", get_models_dir()),
            ("configs_dir", get_configs_dir()),
            ("runs_dir", get_runs_dir()),
            ("logs_path", get_logs_path()),
            ("task_path", get_task_path("univariate/test_univariate")),
            ("model_path", get_model_path("deterministic", "prophet")),
        ]
        
        for name, path in paths:
            print(f"  {name}: {path} (absolute: {path.is_absolute()})")
            assert path.is_absolute(), f"{name} should be an absolute path"
    
    def test_ensure_directory_exists(self):
        """Test that ensure_directory_exists creates directories properly."""
        test_dir = get_runs_dir() / "test_temp_directory"
        
        print(f"\nEnsure Directory Exists:")
        print(f"  Test directory: {test_dir}")
        
        # Ensure it doesn't exist first
        if test_dir.exists():
            test_dir.rmdir()
        
        # Test creation
        ensure_directory_exists(test_dir)
        assert test_dir.exists(), "Directory should be created"
        assert test_dir.is_dir(), "Path should be a directory"
        print(f"  ✓ Directory created successfully")
        
        # Test idempotency (should not raise error if exists)
        ensure_directory_exists(test_dir)
        assert test_dir.exists(), "Directory should still exist"
        print(f"  ✓ Idempotent operation successful")
        
        # Cleanup
        test_dir.rmdir()
        print(f"  ✓ Cleanup successful")
    
    def test_all_paths_summary(self):
        """Print a summary of all generated paths."""
        print(f"\n" + "="*80)
        print(f"SUMMARY OF ALL GENERATED PATHS")
        print(f"="*80)
        
        print(f"\n1. Project Root:")
        print(f"   {get_project_root()}")
        
        print(f"\n2. Tasks Directory:")
        print(f"   {get_tasks_dir()}")
        
        print(f"\n3. Models Directory:")
        print(f"   {get_models_dir()}")
        
        print(f"\n4. Configs Directory:")
        print(f"   {get_configs_dir()}")
        
        print(f"\n5. Runs Directory:")
        print(f"   {get_runs_dir()}")
        
        print(f"\n6. Logs Directory:")
        print(f"   {get_logs_path()}")
        
        print(f"\n7. Example Task Paths:")
        examples = [
            "multivariate/baggage_100_multivariate",
            "univariate/daily_gpu_price_univariate",
            "multivariate/test_multivariate",
        ]
        for example in examples:
            print(f"   Task '{example}': {get_task_path(example)}")
        
        print(f"\n8. Example Model Paths:")
        model_examples = [
            ("deterministic", "arima"),
            ("deterministic", "prophet"),
            ("stochastic", "chronos"),
            ("stochastic", "deepar"),
        ]
        for model_type, model_name in model_examples:
            print(f"   Model '{model_type}/{model_name}': {get_model_path(model_type, model_name)}")
        
        print(f"\n" + "="*80)
        print(f"ALL PATHS VERIFIED")
        print(f"="*80)

