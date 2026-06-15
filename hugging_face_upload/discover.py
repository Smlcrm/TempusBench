"""
Discover TempusBench task assets for Hugging Face upload and local validation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

HF_TASKS_PREFIX = "tasks"
CATEGORIES = ("univariate", "multivariate", "covariate")
DEFAULT_REPO_ID = "Smlcrm/tempus_bench_tasks"


@dataclass(frozen=True)
class TaskAsset:
    """One task CSV referenced by a local task.yaml."""

    category: str
    folder_name: str
    file_name: str
    local_csv_path: Path
    hf_path: str


def resolve_tasks_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            sys.exit(f"Not a directory: {path}")
        return path

    here = Path(__file__).resolve().parent
    for base in (here, *here.parents):
        candidate = base / "tempus_bench" / "tasks"
        if candidate.is_dir():
            return candidate

    cwd = Path.cwd().resolve() / "tempus_bench" / "tasks"
    if cwd.is_dir():
        return cwd

    sys.exit(
        "Could not find tempus_bench/tasks. Pass --tasks-dir PATH "
        "or run from the repository root."
    )


def _load_task_block(task_yaml: Path) -> dict:
    with task_yaml.open(encoding="utf-8") as handle:
        docs = list(yaml.safe_load_all(handle))
    for doc in docs:
        if doc and isinstance(doc, dict) and "task" in doc:
            return doc["task"]
    sys.exit(f"No 'task:' block found in {task_yaml}")


def discover_task_csvs(tasks_root: Path) -> list[TaskAsset]:
    """Return one TaskAsset per task folder that has task.yaml + CSV."""
    assets: list[TaskAsset] = []

    for category in CATEGORIES:
        cat_dir = tasks_root / category
        if not cat_dir.is_dir():
            continue
        for task_dir in sorted(cat_dir.iterdir(), key=lambda p: p.name.lower()):
            if not task_dir.is_dir():
                continue
            task_yaml = task_dir / "task.yaml"
            if not task_yaml.is_file():
                continue

            block = _load_task_block(task_yaml)
            dataset = block.get("dataset") or {}
            file_name = dataset.get("file_name") or block.get("file_name")
            if not file_name:
                sys.exit(f"Missing dataset.file_name in {task_yaml}")

            local_csv = (task_dir / file_name).resolve()
            if not local_csv.is_file():
                sys.exit(f"CSV missing for task {task_dir.name}: {local_csv}")

            folder = task_dir.name
            hf_path = f"{HF_TASKS_PREFIX}/{category}/{folder}/{file_name}".replace("\\", "/")
            assets.append(
                TaskAsset(
                    category=category,
                    folder_name=folder,
                    file_name=file_name,
                    local_csv_path=local_csv,
                    hf_path=hf_path,
                )
            )

    assets.sort(key=lambda a: (a.category, a.folder_name))
    return assets


def validate_local_tasks_complete(tasks_root: Path) -> list[str]:
    """
    Return error messages for an incomplete local tasks tree (empty if OK).
    """
    errors: list[str] = []
    if not tasks_root.is_dir():
        return [f"Tasks directory not found: {tasks_root}"]

    for category in CATEGORIES:
        cat_dir = tasks_root / category
        if not cat_dir.is_dir():
            errors.append(f"Missing category directory: {cat_dir}")
            continue

    try:
        assets = discover_task_csvs(tasks_root)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(str(exc))
        return errors

    if not assets:
        errors.append("No task CSV assets discovered (check task.yaml files).")
        return errors

    expected_per_category = {c: 0 for c in CATEGORIES}
    for asset in assets:
        expected_per_category[asset.category] += 1

    for category, count in expected_per_category.items():
        if count == 0:
            errors.append(f"No tasks found under {tasks_root / category}")

    return errors
