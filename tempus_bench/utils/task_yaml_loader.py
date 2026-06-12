"""Load tasks task.yaml files into TaskConfig objects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from tempus_bench.utils.configs import TaskConfig

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
    return TaskConfig(
        **_base_fields(
            raw,
            task_name=task_name,
            task_path=task_path,
            task_mode="univariate",
        ),
        target_variable_names=target_names,
        covariate_variable_names=[],
    )


def _build_multivariate_config(raw: dict, *, task_name: str, task_path: str) -> TaskConfig:
    all_names = list(raw["target_variable_names"])
    return TaskConfig(
        **_base_fields(
            raw,
            task_name=task_name,
            task_path=task_path,
            task_mode="multivariate",
        ),
        target_variable_names=all_names,
        covariate_variable_names=[],
    )


def _build_covariate_config(raw: dict, *, task_name: str, task_path: str) -> TaskConfig:
    covariate_target = raw["target_variable_name"]
    covariate_names = list(raw.get("covariate_variable_names") or [])
    if covariate_target in covariate_names:
        raise ValueError(
            f"{task_name}: target_variable_name must not appear in covariate_variable_names"
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
    )


def build_task_config(task_dir: Path) -> TaskConfig:
    """Build a TaskConfig from a task directory (folder basename = task_name)."""
    task_dir = task_dir.resolve()
    task_path = str(task_dir)
    task_name = task_dir.name
    raw = _read_task_documents(task_dir)[0]
    kind = _task_kind(task_dir)

    if kind == "univariate":
        return _build_univariate_config(raw, task_name=task_name, task_path=task_path)
    if kind == "multivariate":
        return _build_multivariate_config(raw, task_name=task_name, task_path=task_path)
    if kind == "covariate":
        return _build_covariate_config(raw, task_name=task_name, task_path=task_path)
    raise ValueError(f"Unsupported task folder kind {kind!r} under {task_dir}")


def build_task_configs(logical_name: str, task_dir: Path) -> list[TaskConfig]:
    """Backward-compatible wrapper returning a single TaskConfig."""
    task_dir = task_dir.resolve()
    if logical_name != task_dir.name:
        raise ValueError(
            f"Logical task name {logical_name!r} does not match folder {task_dir.name!r}"
        )
    return [build_task_config(task_dir)]


def load_task_config_from_task_dir(task_dir: Path) -> TaskConfig:
    """Load a single TaskConfig from a task directory."""
    return build_task_config(task_dir.resolve())
