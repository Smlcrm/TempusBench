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


# TODO: REMOVE
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


def get_available_models() -> set:
    """
    Get all available model names from the models directory structure.

    This function scans the models directory to find all model folders that contain
    a corresponding model file. The model file naming convention is {model_name}_model.py.

    Returns:
        set: Set of available model names found in the models directory
    """
    available_models = set()
    models_dir = get_models_dir()
    # Look for model folders directly in the models directory
    for model_folder in models_dir.iterdir():
        if (
            model_folder.is_dir()
            and not model_folder.name.startswith("__")
            and not model_folder.name.startswith(".")
        ):
            # Check if it has a model file
            model_file = model_folder / f"{model_folder.name}_model.py"
            if model_file.exists():
                available_models.add(model_folder.name)

    return available_models


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
    dataset_path = Path(get_task_path(task_name)) / (task_name + ".csv")
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


def find_task_directories(task_path_pattern: str) -> dict[str, str]:
    """
    Find task directories based on a task path pattern.

    This function searches for task directories that match the specified pattern.
    Supported patterns include:
    - "*": All task directories
    - "univariate/*": All univariate task directories
    - "multivariate/*": All multivariate task directories
    - "specific_task": A specific task directory (e.g., "univariate/specific_task")

    Args:
        task_path_pattern: Pattern to match task directories against

    Returns:
        Dict[str, str]: Mapping from task names to absolute directory paths that match the pattern.

    Example:
        >>> find_task_directories("univariate/*")
        {'task1': '/path/to/tasks/univariate/task1', 'task2': '/path/to/tasks/univariate/task2'}
    """
    tasks_dir = get_tasks_dir()
    task_paths = {}

    pattern = task_path_pattern

    if pattern == "*":
        # Find all task directories
        for subdir in tasks_dir.iterdir():
            if subdir.is_dir():
                for task_path in subdir.iterdir():
                    if task_path.is_dir():
                        task_paths[task_path.name] = str(task_path)
    elif pattern.endswith("/*"):
        # Find directories in specific subdirectory
        subdir_name = pattern[:-2]
        subdir_path = tasks_dir / subdir_name
        if subdir_path.exists():
            for task_path in subdir_path.iterdir():
                if task_path.is_dir():
                    task_paths[task_path.name] = str(task_path)
    else:
        # Specific task directory
        task_path = tasks_dir / pattern
        if task_path.exists():
            task_paths[task_path.name] = str(task_path)

    return task_paths


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)
