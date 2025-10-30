"""
Path utilities for inferring absolute paths from the project structure.

This module provides utilities to infer absolute paths based on the fixed
project structure, eliminating the need for directory path configuration.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.

    Returns:
        Path: Absolute path to the project root
    """
    # Get the directory containing this file (tempus_bench/utils/)
    current_file = Path(__file__).resolve()
    # Go up two levels to reach project root
    return current_file.parent.parent.parent


def get_tasks_dir() -> Path:
    """
    Get the absolute path to the tasks directory.

    Returns:
        Path: Absolute path to tempus_bench/tasks/

    Raises:
        FileNotFoundError: If the tasks directory doesn't exist
    """
    tasks_dir = get_project_root() / "tempus_bench" / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    return tasks_dir


def get_models_dir() -> Path:
    """
    Get the absolute path to the models directory.

    Returns:
        Path: Absolute path to tempus_bench/models/

    Raises:
        FileNotFoundError: If the models directory doesn't exist
    """
    models_dir = get_project_root() / "tempus_bench" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    return models_dir


def get_configs_dir() -> Path:
    """
    Get the absolute path to the configs directory.

    Returns:
        Path: Absolute path to tempus_bench/config/

    Raises:
        FileNotFoundError: If the configs directory doesn't exist
    """
    configs_dir = get_project_root() / "tempus_bench" / "config"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")
    return configs_dir


def get_absolute_runs_dir(runs_dir_relative: str) -> Path:
    """
    Get the absolute path to the runs directory from a relative path.

    Args:
        runs_dir_relative: Relative path to runs directory (e.g., "runs")

    Returns:
        Path: Absolute path to runs directory
    """
    runs_path = Path(runs_dir_relative)
    if runs_path.is_absolute():
        return runs_path
    return get_project_root() / runs_dir_relative


def get_task_path(task_name: str) -> Path:
    """
    Get the absolute path to a specific task directory.

    Args:
        task_name: Name of the task (e.g., 'multivariate/baggage_100_multivariate')

    Returns:
        Path: Absolute path to the task directory

    Raises:
        FileNotFoundError: If the task directory doesn't exist
    """
    task_path = get_tasks_dir() / task_name
    if not task_path.exists():
        raise FileNotFoundError(f"Task directory not found: {str(task_path)}")
    return task_path


def get_dataset_path(task_name: str) -> Path:
    """
    Get the absolute path to a specific dataset file.

    Args:
        task_name: Name of the task (e.g., 'baggage_100_multivariate')

    Returns:
        Path: Absolute path to the dataset file

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    dataset_path = Path(get_task_path(task_name)) / (task_name + '.csv')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return dataset_path


def get_model_path(model_type: str, model_name: str) -> Path:
    """
    Get the absolute path to a specific model directory.

    Args:
        model_type: Type of model ('deterministic', 'stochastic', or 'hybrid') - deprecated, kept for backwards compatibility
        model_name: Name of the model

    Returns:
        Path: Absolute path to the model directory

    Raises:
        FileNotFoundError: If the model directory doesn't exist
        
    Note:
        The model_type parameter is deprecated as models are now organized by their
        settings.yaml file rather than folder structure. It's kept for backwards compatibility.
    """
    # Models are now directly in the models directory, not in subdirectories
    model_path = get_models_dir() / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    return model_path


def get_runs_dir() -> Path:
    """
    Get the absolute path to the runs directory.

    Returns:
        Path: Absolute path to runs directory

    Raises:
        FileNotFoundError: If the runs directory doesn't exist
    """
    runs_dir = get_project_root() / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    return runs_dir


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)
