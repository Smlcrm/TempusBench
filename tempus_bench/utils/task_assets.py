"""
Ensure TempusBench task data is present locally by downloading from Hugging Face.

Task CSVs (and optionally the full ``tempus_bench/tasks/`` tree) are hosted at
``Smlcrm/tempus_bench_tasks`` so the git repository stays lightweight.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from tempus_bench.utils.paths import get_project_root

TASKS_REPO_ID = os.environ.get("TEMPUS_BENCH_TASKS_REPO_ID", "Smlcrm/tempus_bench_tasks")
SKIP_DOWNLOAD_ENV = "TEMPUS_BENCH_SKIP_TASK_DOWNLOAD"
HF_TASKS_PREFIX = "tasks"
CATEGORIES = ("univariate", "multivariate", "covariate")

_ensured = False


@dataclass(frozen=True)
class TaskAssetSpec:
    """One task file expected under ``tempus_bench/tasks/``."""

    category: str
    folder_name: str
    file_name: str
    local_path: Path
    hf_path: str  # repo-relative posix path


def _tasks_root() -> Path:
    return get_project_root() / "tempus_bench" / "tasks"


def _load_task_block(task_yaml: Path) -> dict:
    with task_yaml.open(encoding="utf-8") as handle:
        docs = list(yaml.safe_load_all(handle))
    for doc in docs:
        if doc and isinstance(doc, dict) and "task" in doc:
            return doc["task"]
    raise ValueError(f"No 'task:' block found in {task_yaml}")


def discover_expected_assets(tasks_root: Path | None = None) -> list[TaskAssetSpec]:
    """
    Discover per-task CSV assets from local ``task.yaml`` files.

    Returns an empty list when no task definitions exist locally (fresh clone).
    """
    root = tasks_root or _tasks_root()
    specs: list[TaskAssetSpec] = []

    if not root.is_dir():
        return specs

    for category in CATEGORIES:
        cat_dir = root / category
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
            file_name = dataset.get("file_name")
            if not file_name:
                file_name = block.get("file_name")
            if not file_name:
                raise ValueError(f"Missing dataset.file_name in {task_yaml}")

            folder = task_dir.name
            local_csv = task_dir / file_name
            hf_path = f"{HF_TASKS_PREFIX}/{category}/{folder}/{file_name}".replace("\\", "/")
            specs.append(
                TaskAssetSpec(
                    category=category,
                    folder_name=folder,
                    file_name=file_name,
                    local_path=local_csv,
                    hf_path=hf_path,
                )
            )
    return specs


def _missing_assets(specs: list[TaskAssetSpec]) -> list[TaskAssetSpec]:
    return [spec for spec in specs if not spec.local_path.is_file()]


def _skip_download_enabled() -> bool:
    return os.environ.get(SKIP_DOWNLOAD_ENV, "").strip().lower() in {"1", "true", "yes"}


def _download_file(repo_id: str, hf_path: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_path,
        repo_type="dataset",
    )
    shutil.copy2(cached, dest)


def _download_snapshot(repo_id: str, tasks_root: Path) -> None:
    from huggingface_hub import snapshot_download

    project_root = get_project_root()
    package_root = project_root / "tempus_bench"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(package_root),
        allow_patterns=[f"{HF_TASKS_PREFIX}/**"],
    )
    if not tasks_root.is_dir():
        # Recover from older downloads that used project_root as local_dir.
        misplaced = project_root / HF_TASKS_PREFIX
        if misplaced.is_dir():
            package_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(misplaced), str(tasks_root))
    if not tasks_root.is_dir():
        raise FileNotFoundError(
            f"Download completed but tasks directory is still missing: {tasks_root}"
        )


def ensure_task_assets(*, force: bool = False) -> None:
    """
    Ensure task CSV files exist under ``tempus_bench/tasks/``.

    When files are missing, downloads them from the public Hugging Face dataset
    with an explanatory message and a progress bar.

    Args:
        force: Re-check even if a prior call succeeded in this process.
    """
    global _ensured
    if _ensured and not force:
        return

    tasks_root = _tasks_root()
    specs = discover_expected_assets(tasks_root)
    missing = _missing_assets(specs)

    if not missing and tasks_root.is_dir() and specs:
        _ensured = True
        return

    if _skip_download_enabled():
        if not tasks_root.is_dir() or missing:
            missing_paths = [str(s.local_path) for s in missing]
            raise FileNotFoundError(
                "Task data is not available locally and automatic download is disabled "
                f"({SKIP_DOWNLOAD_ENV} is set). Missing: {missing_paths or [str(tasks_root)]}. "
                f"Unset {SKIP_DOWNLOAD_ENV} or run: "
                f"python hugging_face_upload/upload_dataset.py --repo-id {TASKS_REPO_ID}"
            )
        _ensured = True
        return

    repo_id = TASKS_REPO_ID
    dataset_url = f"https://huggingface.co/datasets/{repo_id}"

    if not specs:
        print(
            f"\n[TempusBench] Task definitions not found under {tasks_root}.\n"
            f"  Task data is hosted on Hugging Face to keep the repository lightweight.\n"
            f"  Dataset: {dataset_url}\n"
            f"  Downloading the tasks folder now...\n",
            flush=True,
        )
        _download_snapshot(repo_id, tasks_root)
        specs = discover_expected_assets(tasks_root)
        missing = _missing_assets(specs)
        if missing:
            missing_list = ", ".join(s.hf_path for s in missing)
            raise FileNotFoundError(
                f"Tasks folder downloaded but CSV assets are still missing: {missing_list}"
            )
        print("[TempusBench] Task data download complete.\n", flush=True)
        _ensured = True
        return

    if not missing:
        _ensured = True
        return

    print(
        f"\n[TempusBench] Task CSV data is hosted on Hugging Face ({repo_id}) "
        f"to keep the git repository lightweight.\n"
        f"  Dataset: {dataset_url}\n"
        f"  {len(missing)} of {len(specs)} task CSV file(s) missing locally; downloading now...\n",
        flush=True,
    )

    for spec in tqdm(missing, desc="Downloading task CSVs", unit="file", file=sys.stdout):
        _download_file(repo_id, spec.hf_path, spec.local_path)

    still_missing = _missing_assets(specs)
    if still_missing:
        paths = ", ".join(str(s.local_path) for s in still_missing)
        raise FileNotFoundError(f"Failed to download task CSV(s): {paths}")

    print("[TempusBench] Task data download complete.\n", flush=True)
    _ensured = True


def reset_ensure_cache() -> None:
    """Clear the in-process ensure cache (for tests)."""
    global _ensured
    _ensured = False
