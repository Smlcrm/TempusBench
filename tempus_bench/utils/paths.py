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
    """
    return get_project_root() / "tempus_bench" / "tasks"


def get_models_dir() -> Path:
    """
    Get the absolute path to the models directory.

    Returns:
        Path: Absolute path to tempus_bench/models/
    """
    return get_project_root() / "tempus_bench" / "models"


def get_configs_dir() -> Path:
    """
    Get the absolute path to the configs directory.

    Returns:
        Path: Absolute path to tempus_bench/config/
    """
    return get_project_root() / "tempus_bench" / "config"


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
    """
    return get_tasks_dir() / task_name


def get_dataset_path(task_name: str) -> Path:
    """
    Get the absolute path to a specific dataset file.

    Args:
        task_name: Name of the task (e.g., 'baggage_100_multivariate')

    Returns:
        Path: Absolute path to the dataset file
    """
    return Path(get_task_path(task_name)) / (task_name + '.csv')


def get_model_path(model_type: str, model_name: str) -> Path:
    """
    Get the absolute path to a specific model directory.

    Args:
        model_type: Type of model ('deterministic' or 'stochastic')
        model_name: Name of the model

    Returns:
        Path: Absolute path to the model directory
    """
    return get_models_dir() / model_type / model_name


def get_runs_dir() -> Path:
    """
    Get the absolute path to the runs directory.

    Returns:
        Path: Absolute path to runs directory
    """
    return get_project_root() / "runs"


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)
