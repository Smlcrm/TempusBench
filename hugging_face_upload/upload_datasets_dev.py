"""
Upload the repo-root Datasets/ tree to a Hugging Face dataset repo on the dev branch.

Does not upload to main. Creates the dev branch if missing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_DATASETS_PREFIX = "Datasets"
DEFAULT_REPO_ID = "Smlcrm/tempus_bench_tasks"
DEFAULT_BRANCH = "dev"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _staging_root() -> Path:
    return Path(__file__).resolve().parent / ".staging_datasets_dev"


def _collect_dataset_files(datasets_root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for path in sorted(datasets_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        rel = path.relative_to(datasets_root)
        pairs.append((path, f"{HF_DATASETS_PREFIX}/{rel.as_posix()}"))
    return pairs


def _validate_datasets_root(datasets_root: Path) -> list[str]:
    errors: list[str] = []
    if not datasets_root.is_dir():
        return [f"Datasets root not found: {datasets_root}"]

    dataset_dirs = [
        p
        for category in datasets_root.iterdir()
        if category.is_dir()
        for p in category.iterdir()
        if p.is_dir()
    ]
    if not dataset_dirs:
        errors.append(f"No dataset folders under {datasets_root}")

    for task_dir in dataset_dirs:
        meta = task_dir / "metadata.json"
        csvs = list(task_dir.glob("*.csv"))
        if not meta.is_file():
            errors.append(f"Missing metadata.json: {task_dir}")
        if len(csvs) != 1:
            errors.append(f"Expected one CSV in {task_dir}, found {len(csvs)}")
        elif csvs[0].stem != task_dir.name:
            errors.append(
                f"CSV name mismatch in {task_dir}: {csvs[0].name} vs {task_dir.name}.csv"
            )
    return errors


def _render_readme(dataset_count: int, category_count: int) -> str:
    return f"""---
license: mit
task_categories:
- time-series-forecasting
language:
- en
tags:
- tempus-bench
- forecasting
- time-series
---

# TempusBench Datasets (dev branch)

This branch contains the TempusBench **Datasets/** layout:

```
Datasets/
  {{dataset_category}}/
    {{dataset_name}}/
      metadata.json
      {{dataset_name}}.csv
```

- **Categories:** {category_count}
- **Datasets:** {dataset_count}

Task definitions live in the git repository under `Tasks/Application Tasks/` and are not uploaded here.
"""


def prepare_staging(datasets_root: Path, staging: Path) -> tuple[int, int]:
    errors = _validate_datasets_root(datasets_root)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    categories = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    dataset_count = sum(
        1
        for category in datasets_root.iterdir()
        if category.is_dir()
        for task_dir in category.iterdir()
        if task_dir.is_dir()
    )

    if staging.exists():
        shutil.rmtree(staging)

    pairs = _collect_dataset_files(datasets_root)
    for local_path, repo_rel in pairs:
        dest = staging / repo_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    (staging / "README.md").write_text(
        _render_readme(dataset_count, len(categories)),
        encoding="utf-8",
    )
    return dataset_count, len(categories)


def ensure_branch(api: HfApi, repo_id: str, branch: str, *, token: str | None) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=token,
    )
    try:
        api.create_branch(
            repo_id=repo_id,
            branch=branch,
            repo_type="dataset",
            token=token,
        )
        print(f"Created branch: {branch}")
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "exist" in message:
            print(f"Branch already exists: {branch}")
        else:
            raise


def upload(staging: Path, repo_id: str, branch: str, *, token: str | None) -> None:
    api = HfApi(token=token)
    ensure_branch(api, repo_id, branch, token=token)
    api.upload_folder(
        folder_path=str(staging),
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        token=token,
        commit_message=f"Upload Datasets/ layout to {branch}",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}/tree/{branch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=_repo_root() / "Datasets",
        help="Path to repo-root Datasets/",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=_staging_root(),
        help="Local staging directory",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("TEMPUS_BENCH_TASKS_REPO_ID", DEFAULT_REPO_ID),
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Target dataset branch (default: dev; never main)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF API token",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only populate staging; do not upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned upload paths only",
    )
    args = parser.parse_args()

    if args.branch.lower() == "main":
        sys.exit("Refusing to upload to main. Use --branch dev or another non-main branch.")

    datasets_root = args.datasets_dir.resolve()
    staging = args.staging.resolve()

    if args.dry_run:
        errors = _validate_datasets_root(datasets_root)
        if errors:
            for msg in errors:
                print(f"[dry-run] ERROR: {msg}")
            sys.exit(1)
        pairs = _collect_dataset_files(datasets_root)
        print(f"[dry-run] Would stage {len(pairs)} file(s) for {args.repo_id}@{args.branch}")
        for local_path, repo_rel in pairs[:8]:
            print(f"  {repo_rel}  <=  {local_path}")
        if len(pairs) > 8:
            print(f"  ... and {len(pairs) - 8} more")
        return

    dataset_count, category_count = prepare_staging(datasets_root, staging)
    print(f"Staged {dataset_count} datasets ({category_count} categories) under {staging}")

    if args.prepare_only:
        print("Done (--prepare-only).")
        return

    upload(staging, args.repo_id, args.branch, token=args.token)


if __name__ == "__main__":
    main()
