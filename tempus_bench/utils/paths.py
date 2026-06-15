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

    Downloads task CSV data from Hugging Face on first use when files are missing.

    Returns:
        Path: Absolute path to tempus_bench/tasks/

    Raises:
        FileNotFoundError: If the tasks directory doesn't exist
    """
    from tempus_bench.utils.task_assets import ensure_task_assets

    ensure_task_assets()
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
            settings_file = model_folder / "settings.yaml"
            if model_file.exists() and settings_file.exists():
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
        task_name: Name of the task (e.g., 'multivariate/multivariate_transport_monthly_airline_baggage_complaints')

    Returns:
        Path: Absolute path to the task directory

    Raises:
        FileNotFoundError: If the task directory doesn't exist
    """
    task_path = get_tasks_dir() / task_name
    if not task_path.exists():
        raise FileNotFoundError(f"Task directory not found: {str(task_path)}")
    return task_path


def get_dataset_path(task_name: str, *, file_name: str | None = None) -> Path:
    """
    Get the absolute path to a task dataset CSV.

    Args:
        task_name: Relative task path (e.g. ``univariate/foo`` or ``multivariate/bar``).
        file_name: CSV filename from task.yaml; when omitted, tries ``{basename}.csv``
            then the sole ``*.csv`` in the task folder.

    Returns:
        Path: Absolute path to the dataset file

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    task_dir = get_task_path(task_name)
    if file_name:
        dataset_path = task_dir / file_name
    else:
        primary = task_dir / f"{task_dir.name}.csv"
        if primary.exists():
            dataset_path = primary
        else:
            csv_files = sorted(task_dir.glob("*.csv"))
            if len(csv_files) == 1:
                dataset_path = csv_files[0]
            else:
                dataset_path = primary
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


def get_logs_path() -> Path:
    """
    Get the absolute path to the project logs directory (may not exist yet).

    Returns:
        Path: Absolute path to ``<project_root>/logs``
    """
    return get_project_root() / "logs"


def find_task_directories(task_path_pattern: str) -> dict[str, str]:
    """
    Find task directories based on a task path pattern under tasks/.

    Supported patterns:
    - ``*``: all folders under univariate/, multivariate/, and covariate/
    - ``univariate/*`` / ``multivariate/*`` / ``covariate/*``: all folders in that category
    - ``univariate/foo`` / ``multivariate/foo`` / ``covariate/foo``
    """
    tasks_dir = get_tasks_dir()
    task_paths: dict[str, str] = {}
    pattern = task_path_pattern.strip()

    def _register_folder(task_path: Path) -> None:
        if task_path.is_dir():
            task_paths[task_path.name] = str(task_path)

    if pattern == "*":
        for subdir_name in ("univariate", "multivariate", "covariate"):
            subdir_path = tasks_dir / subdir_name
            if not subdir_path.is_dir():
                continue
            for task_path in subdir_path.iterdir():
                _register_folder(task_path)
        return task_paths

    if pattern.endswith("/*"):
        subdir_name = pattern[:-2]
        subdir_path = tasks_dir / subdir_name
        if subdir_path.is_dir():
            for task_path in subdir_path.iterdir():
                _register_folder(task_path)
        return task_paths

    task_path = tasks_dir / pattern
    _register_folder(task_path)
    return task_paths


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Path to the directory to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)


def get_available_metrics() -> list[Path]:
    """
    Get all files containing subclasses of the BaseMetric class.

    This function scans the metrics directory for Python files and assumes all files
    (except base_metric.py, __init__.py, and cache files) contain BaseMetric subclasses.

    Returns:
        list[Path]: List of absolute file paths containing BaseMetric subclasses

    Raises:
        FileNotFoundError: If the metrics directory doesn't exist
    """
    # Get the absolute path to the metrics directory
    metrics_dir = get_project_root() / "tempus_bench" / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    metric_files = []

    # Scan all Python files in the metrics directory
    for file_path in metrics_dir.glob("*.py"):
        # Skip files starting with "." or "__", and base_metric.py
        if (
            not file_path.name.startswith(".")
            and not file_path.name.startswith("__")
            and file_path.name != "base_metric.py"
        ):
            metric_files.append(file_path.resolve())

    return metric_files
