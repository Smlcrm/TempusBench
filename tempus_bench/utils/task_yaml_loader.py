"""Load catalog task YAML documents into TaskConfig objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from tempus_bench.utils.configs import TaskConfig

TaskMode = Literal["univariate", "multivariate", "covariate"]


def infer_task_mode(
    target_variable_names: list[str],
    covariate_variable_names: list[str],
) -> TaskMode:
    """Infer evaluation mode from target/covariate lists."""
    n_targets = len(target_variable_names)
    n_covariates = len(covariate_variable_names)
    if n_targets < 1:
        raise ValueError("target_variable_names must contain at least one name")
    if n_covariates > 0:
        return "covariate"
    if n_targets == 1:
        return "univariate"
    return "multivariate"


def build_task_config_from_raw(raw: dict[str, Any]) -> TaskConfig:
    """Build a TaskConfig from a catalog task YAML document."""
    if "target_variable_name" in raw:
        raise ValueError(
            f"Task {raw.get('task_name')!r}: singular 'target_variable_name' is not "
            "allowed; use 'target_variable_names'."
        )
    if "target_variable_names" not in raw:
        raise ValueError(
            f"Task {raw.get('task_name')!r}: missing required 'target_variable_names'."
        )

    task_name = raw["task_name"]
    dataset_category = raw["dataset_category"]
    dataset_name = raw["dataset_name"]
    targets = list(raw["target_variable_names"])
    covariates = list(raw.get("covariate_variable_names") or [])
    task_mode = infer_task_mode(targets, covariates)
    logical_path = f"{dataset_category}/{task_name}"

    return TaskConfig(
        task_name=task_name,
        task_path=logical_path,
        task_description=raw.get("task_description") or "",
        task_catalog=raw.get("task_catalog") or "application",
        dataset_category=dataset_category,
        dataset_name=dataset_name,
        context_window=raw["context_window"],
        forecast_horizon=raw["forecast_horizon"],
        handle_missing=raw.get("handle_missing", "interpolate"),
        normalization_method=raw.get("normalization_method", "standard"),
        file_name=f"{dataset_name}.csv",
        task_mode=task_mode,
        target_variable_names=targets,
        covariate_variable_names=covariates,
    )


def build_task_config(task_dir: Path) -> TaskConfig:
    """
    Legacy entry point kept for older call sites.

    The new layout has no per-task directories. Prefer ``build_task_config_from_raw``.
    """
    raise RuntimeError(
        "Per-folder task directories are no longer supported. "
        f"Received {task_dir}. Use build_task_config_from_raw() with catalog YAML docs."
    )


def build_task_configs(logical_name: str, task_dir: Path) -> list[TaskConfig]:
    """Backward-compatible wrapper; see ``build_task_config``."""
    del logical_name
    return [build_task_config(task_dir)]


def load_task_config_from_task_dir(task_dir: Path) -> TaskConfig:
    """Legacy loader; see ``build_task_config``."""
    return build_task_config(task_dir.resolve())
