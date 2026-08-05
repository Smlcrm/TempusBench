"""
Ensure TempusBench dataset CSVs are present locally by syncing from Hugging Face.

Task definitions live in git under ``Tasks/``. Dataset CSVs are hosted at
``Smlcrm/tempus_bench_tasks`` under ``Datasets/`` and synced into repo-root
``Datasets/`` once at benchmark start.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from tempus_bench.utils.paths import get_project_root, load_all_task_documents

TASKS_REPO_ID = os.environ.get("TEMPUS_BENCH_TASKS_REPO_ID", "Smlcrm/tempus_bench_tasks")
DATASETS_REVISION = os.environ.get("TEMPUS_BENCH_DATASETS_REVISION", "dev")
SKIP_DOWNLOAD_ENV = "TEMPUS_BENCH_SKIP_TASK_DOWNLOAD"
SKIP_DATASET_SYNC_ENV = "TEMPUS_BENCH_SKIP_DATASET_SYNC"
HF_DATASETS_PREFIX = "Datasets"

_ensured = False


@dataclass(frozen=True)
class DatasetAssetSpec:
    """One dataset CSV expected under repo-root ``Datasets/``."""

    dataset_category: str
    dataset_name: str
    local_path: Path
    hf_path: str


def _datasets_root() -> Path:
    return get_project_root() / "Datasets"


def discover_expected_datasets() -> list[DatasetAssetSpec]:
    """Build expected dataset inventory from catalog task YAMLs under ``Tasks/``."""
    root = _datasets_root()
    specs: list[DatasetAssetSpec] = []
    seen: set[tuple[str, str]] = set()

    for doc in load_all_task_documents():
        category = doc["dataset_category"]
        name = doc["dataset_name"]
        key = (category, name)
        if key in seen:
            continue
        seen.add(key)
        local_csv = root / category / name / f"{name}.csv"
        hf_path = f"{HF_DATASETS_PREFIX}/{category}/{name}/{name}.csv".replace("\\", "/")
        specs.append(
            DatasetAssetSpec(
                dataset_category=category,
                dataset_name=name,
                local_path=local_csv,
                hf_path=hf_path,
            )
        )
    return specs


def _missing_assets(specs: list[DatasetAssetSpec]) -> list[DatasetAssetSpec]:
    return [spec for spec in specs if not spec.local_path.is_file()]


def _skip_sync_enabled() -> bool:
    for env_name in (SKIP_DATASET_SYNC_ENV, SKIP_DOWNLOAD_ENV):
        if os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def _list_remote_dataset_csv_paths(repo_id: str, revision: str) -> set[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    items = api.list_repo_tree(
        repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
    )
    paths: set[str] = set()
    for item in items:
        path = getattr(item, "path", None) or str(item)
        if path.endswith(".csv") and path.startswith(f"{HF_DATASETS_PREFIX}/"):
            paths.add(path.replace("\\", "/"))
    return paths


def _download_file(repo_id: str, revision: str, hf_path: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_path,
        repo_type="dataset",
        revision=revision,
    )
    shutil.copy2(cached, dest)


def ensure_dataset_assets(*, force: bool = False) -> None:
    """
    Ensure expected dataset CSVs exist under repo-root ``Datasets/``.

    Once per process (unless ``force``):
    1. Build expected inventory from ``Tasks/``
    2. Compare to local ``Datasets/``
    3. Auto-download missing files from HF
    4. Fail if HF is unreachable and any expected CSV is missing
    """
    global _ensured
    if _ensured and not force:
        return

    specs = discover_expected_datasets()
    if not specs:
        raise FileNotFoundError(
            "No task documents found under Tasks/. Add catalog YAML files before running."
        )

    missing = _missing_assets(specs)
    if not missing:
        _warn_extra_local(specs)
        _ensured = True
        return

    if _skip_sync_enabled():
        missing_paths = [str(s.local_path) for s in missing]
        raise FileNotFoundError(
            "Dataset CSVs are missing locally and automatic sync is disabled "
            f"({SKIP_DATASET_SYNC_ENV}/{SKIP_DOWNLOAD_ENV}). Missing: {missing_paths}."
        )

    repo_id = TASKS_REPO_ID
    revision = DATASETS_REVISION
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"

    print(
        f"\n[TempusBench] Dataset CSVs are hosted on Hugging Face ({repo_id}@{revision}).\n"
        f"  Dataset: {dataset_url}\n"
        f"  {len(missing)} of {len(specs)} CSV file(s) missing locally; syncing now...\n",
        flush=True,
    )

    try:
        remote_csvs = _list_remote_dataset_csv_paths(repo_id, revision)
    except Exception as exc:
        missing_paths = [str(s.local_path) for s in missing]
        raise FileNotFoundError(
            "Hugging Face is unreachable and required dataset CSVs are missing locally. "
            f"Missing: {missing_paths}. Error: {exc}"
        ) from exc

    expected_hf = {s.hf_path for s in specs}
    remote_missing = sorted(expected_hf - remote_csvs)
    if remote_missing:
        raise FileNotFoundError(
            f"Remote dataset inventory is incomplete on {repo_id}@{revision}. "
            f"Missing on Hub: {remote_missing}"
        )

    for spec in tqdm(missing, desc="Downloading dataset CSVs", unit="file", file=sys.stdout):
        try:
            _download_file(repo_id, revision, spec.hf_path, spec.local_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to download {spec.hf_path} from {repo_id}@{revision}: {exc}"
            ) from exc

    still_missing = _missing_assets(specs)
    if still_missing:
        paths = ", ".join(str(s.local_path) for s in still_missing)
        raise FileNotFoundError(f"Failed to download dataset CSV(s): {paths}")

    _warn_extra_local(specs)
    print("[TempusBench] Dataset sync complete.\n", flush=True)
    _ensured = True


def _warn_extra_local(specs: list[DatasetAssetSpec]) -> None:
    root = _datasets_root()
    if not root.is_dir():
        return
    expected = {s.local_path.resolve() for s in specs}
    extras: list[str] = []
    for path in root.rglob("*.csv"):
        if path.resolve() not in expected:
            extras.append(str(path.relative_to(root)))
    if extras:
        warnings.warn(
            "Local Datasets/ contains CSV files not referenced by Tasks/: "
            + ", ".join(sorted(extras)[:20])
            + (" ..." if len(extras) > 20 else ""),
            stacklevel=2,
        )


def ensure_task_assets(*, force: bool = False) -> None:
    """Backward-compatible alias for ``ensure_dataset_assets``."""
    ensure_dataset_assets(force=force)


def reset_ensure_cache() -> None:
    """Clear the in-process ensure cache (for tests)."""
    global _ensured
    _ensured = False
