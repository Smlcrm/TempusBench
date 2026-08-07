"""
Path utilities for inferring absolute paths from the project structure.

Discovers catalog tasks under repo-root ``Tasks/`` and resolves dataset CSVs
under repo-root ``Datasets/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.

    Returns:
        Path: Absolute path to the project root
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_tasks_dir() -> Path:
    """
    Get the absolute path to the catalog task definitions directory.

    Returns:
        Path: Absolute path to repo-root ``Tasks/``

    Raises:
        FileNotFoundError: If the Tasks directory doesn't exist
    """
    tasks_dir = get_project_root() / "Tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    return tasks_dir


def get_datasets_dir(*, ensure: bool = True) -> Path:
    """
    Get the absolute path to the local Datasets directory.

    When ``ensure`` is True, syncs missing CSVs from Hugging Face once per process.

    Returns:
        Path: Absolute path to repo-root ``Datasets/``
    """
    if ensure:
        from tempus_bench.utils.task_assets import ensure_dataset_assets

        ensure_dataset_assets()
    return get_project_root() / "Datasets"


def list_task_catalog_dirs() -> list[Path]:
    """Return immediate catalog directories under ``Tasks/`` (e.g. Application Tasks)."""
    tasks_dir = get_tasks_dir()
    return sorted(
        p
        for p in tasks_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("__")
    )


def _iter_category_yaml_files() -> list[Path]:
    files: list[Path] = []
    for catalog_dir in list_task_catalog_dirs():
        files.extend(sorted(catalog_dir.glob("*.yaml")))
        files.extend(sorted(catalog_dir.glob("*.yml")))
    return files


def load_all_task_documents() -> list[dict[str, Any]]:
    """
    Load every ``task:`` document from all catalog YAML files under ``Tasks/``.

    Raises:
        ValueError: On duplicate human ``task_name`` values across the catalog.
        ValueError: If a document uses singular ``target_variable_name``.
    """
    docs: list[dict[str, Any]] = []
    seen_names: dict[str, str] = {}

    for yaml_path in _iter_category_yaml_files():
        with yaml_path.open(encoding="utf-8") as handle:
            raw_docs = list(yaml.safe_load_all(handle))
        for doc in raw_docs:
            if not doc or "task" not in doc:
                continue
            task = doc["task"]
            if not isinstance(task, dict):
                raise ValueError(f"Invalid task document in {yaml_path}")
            if "target_variable_name" in task:
                raise ValueError(
                    f"Singular 'target_variable_name' is not allowed in {yaml_path} "
                    f"(task_name={task.get('task_name')!r}). Use only 'target_variable_names'."
                )
            name = task.get("task_name")
            if not name or not isinstance(name, str):
                raise ValueError(f"Missing task_name in {yaml_path}")
            if name in seen_names:
                raise ValueError(
                    f"Duplicate task_name {name!r}: seen in {seen_names[name]} and {yaml_path}"
                )
            seen_names[name] = str(yaml_path)
            enriched = dict(task)
            enriched["_source_yaml"] = str(yaml_path)
            docs.append(enriched)
    return docs


def find_task_documents(task_path_pattern: str) -> dict[str, dict[str, Any]]:
    """
    Find task documents matching a selector under ``Tasks/``.

    Supported patterns:
    - ``*``: all tasks in every catalog
    - ``{dataset_category}/*``: all tasks in that category
    - ``{dataset_category}/{human task_name}``: one exact task

    Returns:
        Mapping of human ``task_name`` -> raw task document dict.
    """
    pattern = task_path_pattern.strip()
    all_docs = load_all_task_documents()
    matched: dict[str, dict[str, Any]] = {}

    if pattern == "*":
        for doc in all_docs:
            matched[doc["task_name"]] = doc
        return matched

    if pattern.endswith("/*"):
        category = pattern[:-2]
        for doc in all_docs:
            if doc.get("dataset_category") == category:
                matched[doc["task_name"]] = doc
        return matched

    if "/" not in pattern:
        raise ValueError(
            f"Invalid task selector {pattern!r}. Expected '*', "
            "'{category}/*', or '{category}/{task_name}'."
        )

    category, _, task_name = pattern.partition("/")
    for doc in all_docs:
        if doc.get("dataset_category") == category and doc.get("task_name") == task_name:
            matched[doc["task_name"]] = doc
            return matched
    return matched


def find_task_directories(task_path_pattern: str) -> dict[str, str]:
    """
    Backward-compatible discovery API.

    Returns mapping of human ``task_name`` -> logical selector
    ``{dataset_category}/{task_name}``. Prefer ``find_task_documents`` for
    loading full YAML payloads.
    """
    docs = find_task_documents(task_path_pattern)
    return {
        name: f"{doc['dataset_category']}/{doc['task_name']}"
        for name, doc in docs.items()
    }


def get_dataset_path(
    dataset_category: str,
    dataset_name: str,
    *,
    ensure: bool = True,
) -> Path:
    """
    Resolve ``Datasets/{dataset_category}/{dataset_name}/{dataset_name}.csv``.

    Raises:
        FileNotFoundError: If the CSV is missing after optional HF sync.
    """
    datasets_root = get_datasets_dir(ensure=ensure)
    dataset_path = (
        datasets_root / dataset_category / dataset_name / f"{dataset_name}.csv"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return dataset_path


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

    Returns:
        set: Set of available model names found in the models directory
    """
    available_models = set()
    models_dir = get_models_dir()
    for model_folder in models_dir.iterdir():
        if (
            model_folder.is_dir()
            and not model_folder.name.startswith("__")
            and not model_folder.name.startswith(".")
        ):
            model_file = model_folder / f"{model_folder.name}_model.py"
            settings_file = model_folder / "settings.yaml"
            if model_file.exists() and settings_file.exists():
                available_models.add(model_folder.name)
    return available_models


def get_absolute_runs_dir(runs_dir_relative: str) -> Path:
    """Get the absolute path to the runs directory from a relative path."""
    runs_path = Path(runs_dir_relative)
    if runs_path.is_absolute():
        return runs_path
    return get_project_root() / runs_dir_relative


def get_model_path(model_type: str, model_name: str) -> Path:
    """
    Get the absolute path to a specific model directory.

    Note:
        ``model_type`` is deprecated and ignored.
    """
    del model_type
    model_path = get_models_dir() / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    return model_path


def get_runs_dir() -> Path:
    """Get the absolute path to the runs directory."""
    runs_dir = get_project_root() / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    return runs_dir


def get_logs_path() -> Path:
    """Get the absolute path to the project logs directory (may not exist yet)."""
    return get_project_root() / "logs"


def ensure_directory_exists(path: Path) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def get_available_metrics() -> list[Path]:
    """
    Get all files containing subclasses of the BaseMetric class.

    Returns:
        list[Path]: List of absolute file paths containing BaseMetric subclasses
    """
    metrics_dir = get_project_root() / "tempus_bench" / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    metric_files = []
    for file_path in metrics_dir.glob("*.py"):
        if (
            not file_path.name.startswith(".")
            and not file_path.name.startswith("__")
            and file_path.name != "base_metric.py"
        ):
            metric_files.append(file_path.resolve())
    return metric_files


def task_dataset_filename(task_name: str, base_seed: int | None) -> str:
    """
    Pickle filename for a prepared task dataset.

    Application tasks have no seed and keep the historical unsuffixed name, so
    their cache paths are unchanged. Generator tasks get one pickle per base seed.

    Args:
        task_name (str): Logical task name.
        base_seed (int | None): Base seed for generator tasks, None for application tasks.

    Returns:
        str: The pickle filename.
    """
    if base_seed is None:
        return f"{task_name}.pkl"
    return f"{task_name}__seed{base_seed}.pkl"
