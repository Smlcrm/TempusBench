"""Load tasks task.yaml files into TaskConfig objects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from tempus_bench.utils.configs import TaskConfig

COVARIATE_TASK_SUFFIX = "__covariate"

HandleMissing = Literal[
    "interpolate", "mean", "median", "drop", "forward_fill", "backward_fill"
]
NormalizationMethod = Literal["standard", "none"]
TaskMode = Literal["univariate", "multivariate", "covariate"]


def _task_kind(task_dir: Path) -> str:
    parts = task_dir.resolve().parts
    for marker in ("tasks",):
        try:
            idx = parts.index(marker)
            return parts[idx + 1].lower()
        except (ValueError, IndexError):
            continue
    raise ValueError(f"Cannot infer task kind from path: {task_dir}")


def _read_task_documents(task_dir: Path) -> list[dict]:
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing task.yaml: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as handle:
        documents = list(yaml.safe_load_all(handle))
    payloads = [doc["task"] for doc in documents if doc and "task" in doc]
    if not payloads:
        raise ValueError(f"No valid task document in {yaml_path}")
    return payloads


def _base_fields(raw: dict, *, task_name: str, task_path: str, task_mode: TaskMode) -> dict:
    return {
        "task_name": task_name,
        "task_path": task_path,
        "context_window": raw["context_window"],
        "forecast_horizon": raw["forecast_horizon"],
        "handle_missing": raw.get("handle_missing", "interpolate"),
        "normalization_method": raw.get("normalization_method", "standard"),
        "file_name": raw["file_name"],
        "task_mode": task_mode,
    }


def _build_univariate_config(raw: dict, *, task_name: str, task_path: str) -> TaskConfig:
    target_names = list(raw["target_variable_names"])
    covariate_names = list(raw.get("covariate_variable_names") or [])
    return TaskConfig(
        **_base_fields(
            raw,
            task_name=task_name,
            task_path=task_path,
            task_mode="univariate",
        ),
        target_variable_names=target_names,
        covariate_variable_names=covariate_names,
        multivariate_target_variable_names=None,
        covariate_target_variable_name=None,
    )


def _build_multivariate_config(raw: dict, *, task_name: str, task_path: str) -> TaskConfig:
    all_names = list(raw["multivariate_target_variable_names"])
    return TaskConfig(
        **_base_fields(
            raw,
            task_name=task_name,
            task_path=task_path,
            task_mode="multivariate",
        ),
        target_variable_names=all_names,
        covariate_variable_names=[],
        multivariate_target_variable_names=all_names,
        covariate_target_variable_name=raw.get("covariate_target_variable_name"),
    )


def _build_covariate_config(raw: dict, *, task_name: str, task_path: str) -> TaskConfig:
    covariate_target = raw["covariate_target_variable_name"]
    covariate_names = list(raw.get("covariate_variable_names") or [])
    all_names = list(raw["multivariate_target_variable_names"])
    if covariate_target not in all_names:
        raise ValueError(
            f"{task_name}: covariate_target_variable_name {covariate_target!r} "
            f"not found in multivariate_target_variable_names"
        )
    missing = [name for name in covariate_names if name not in all_names]
    if missing:
        raise ValueError(
            f"{task_name}: covariate_variable_names not in multivariate_target_variable_names: "
            f"{missing[:5]}"
        )
    if covariate_target in covariate_names:
        raise ValueError(
            f"{task_name}: covariate_target_variable_name must not appear in covariate_variable_names"
        )
    return TaskConfig(
        **_base_fields(
            raw,
            task_name=task_name,
            task_path=task_path,
            task_mode="covariate",
        ),
        target_variable_names=[covariate_target],
        covariate_variable_names=covariate_names,
        multivariate_target_variable_names=all_names,
        covariate_target_variable_name=covariate_target,
    )


def build_task_configs(logical_name: str, task_dir: Path) -> list[TaskConfig]:
    """
    Build TaskConfig objects for a discovered logical task name and on-disk folder.

    ``logical_name`` may include ``__covariate`` for covariate-mode logical tasks.
    """
    task_dir = task_dir.resolve()
    task_path = str(task_dir)
    raw = _read_task_documents(task_dir)[0]
    kind = _task_kind(task_dir)
    folder_name = task_dir.name

    if logical_name.endswith(COVARIATE_TASK_SUFFIX):
        expected = f"{folder_name}{COVARIATE_TASK_SUFFIX}"
        if logical_name != expected:
            raise ValueError(
                f"Logical task name {logical_name!r} does not match folder {folder_name!r}"
            )
        if kind != "multivariate":
            raise ValueError(f"Covariate logical task requires multivariate folder: {task_dir}")
        return [_build_covariate_config(raw, task_name=logical_name, task_path=task_path)]

    if logical_name != folder_name:
        raise ValueError(
            f"Logical task name {logical_name!r} does not match folder {folder_name!r}"
        )

    if kind == "univariate":
        return [_build_univariate_config(raw, task_name=logical_name, task_path=task_path)]
    if kind == "multivariate":
        return [_build_multivariate_config(raw, task_name=logical_name, task_path=task_path)]
    raise ValueError(f"Unsupported task folder kind {kind!r} under {task_dir}")


def load_task_config_from_task_dir(
    task_dir: Path,
    *,
    logical_name: str | None = None,
) -> TaskConfig:
    """Load a single TaskConfig from a task directory (defaults to folder basename)."""
    task_dir = task_dir.resolve()
    name = logical_name or task_dir.name
    return build_task_configs(name, task_dir)[0]
