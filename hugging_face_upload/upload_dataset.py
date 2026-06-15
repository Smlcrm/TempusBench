"""
Upload TempusBench task data to a public Hugging Face dataset repository.

Stages the full ``tasks/`` tree (CSVs, task.yaml, metadata.json, catalog files)
from ``tempus_bench/tasks/`` and uploads to ``Smlcrm/tempus_bench_tasks`` by default.

Requires ``HF_TOKEN`` or a prior ``huggingface-cli login``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

import discover as disc
import generate_readme as gr


def _staging_root_default() -> Path:
    return Path(__file__).resolve().parent / ".staging"


def _rel_under_tasks(path: Path, tasks_root: Path) -> str:
    rel = path.relative_to(tasks_root)
    return f"{disc.HF_TASKS_PREFIX}/{rel.as_posix()}"


def _collect_files_to_stage(tasks_root: Path) -> list[tuple[Path, str]]:
    """Return (local_path, repo_relative_path) pairs to upload."""
    pairs: list[tuple[Path, str]] = []

    for path in sorted(tasks_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        pairs.append((path, _rel_under_tasks(path, tasks_root)))

    return pairs


def prepare_staging(tasks_root: Path, staging: Path) -> list[disc.TaskAsset]:
    errors = disc.validate_local_tasks_complete(tasks_root)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    assets = disc.discover_task_csvs(tasks_root)
    if not assets:
        sys.exit("No task CSV assets to upload.")

    if staging.exists():
        shutil.rmtree(staging)

    pairs = _collect_files_to_stage(tasks_root)
    for local_path, repo_rel in pairs:
        dest = staging / repo_rel.replace("/", os.sep)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    readme = gr.render_readme(assets)
    (staging / "README.md").write_text(readme, encoding="utf-8")
    return assets


def ensure_public_dataset(api: HfApi, repo_id: str, *, token: str | None) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=token,
    )
    print(f"Dataset ready (public): https://huggingface.co/datasets/{repo_id}")


def purge_remote_tasks_tree(
    api: HfApi,
    repo_id: str,
    *,
    token: str | None,
    dry_run: bool,
) -> None:
    paths = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    deletes: list[str] = []
    if any(p.startswith(f"{disc.HF_TASKS_PREFIX}/") for p in paths):
        deletes.append(f"{disc.HF_TASKS_PREFIX}/")
    if "README.md" in paths:
        deletes.append("README.md")

    if not deletes:
        print("Remote purge: nothing to remove.")
        return

    if dry_run:
        print(f"[dry-run] Would delete from Hub ({repo_id}): {deletes}")
        return

    api.delete_files(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        delete_patterns=deletes,
        commit_message="Remove previous TempusBench tasks tree before fresh upload",
    )
    print(f"Remote purge: deleted {len(deletes)} pattern(s): {deletes}")


def upload(staging: Path, repo_id: str, *, token: str | None) -> None:
    api = HfApi(token=token)
    ensure_public_dataset(api, repo_id, token=token)
    api.upload_folder(
        folder_path=str(staging),
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        help="Path to tempus_bench/tasks (default: auto-detect)",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help=f"Staging directory (default: {_staging_root_default().as_posix()})",
    )
    parser.add_argument(
        "--repo-id",
        default=disc.DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF API token (default: HF_TOKEN env var or cached CLI login)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only populate staging; do not upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned paths without copying or calling the API",
    )
    parser.add_argument(
        "--purge-remote",
        action="store_true",
        help="Delete remote tasks/ and README.md before uploading",
    )
    args = parser.parse_args()

    tasks_root = disc.resolve_tasks_root(args.tasks_dir)
    staging = args.staging or _staging_root_default()
    api = HfApi(token=args.token)

    if args.dry_run:
        errors = disc.validate_local_tasks_complete(tasks_root)
        if errors:
            for msg in errors:
                print(f"[dry-run] ERROR: {msg}")
            sys.exit(1)
        assets = disc.discover_task_csvs(tasks_root)
        pairs = _collect_files_to_stage(tasks_root)
        print(f"[dry-run] Would stage {len(assets)} task CSV(s) and {len(pairs)} total file(s)")
        for local_path, repo_rel in pairs[:5]:
            print(f"  {repo_rel}  <=  {local_path}")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs) - 5} more")
        if args.purge_remote:
            purge_remote_tasks_tree(api, args.repo_id, token=args.token, dry_run=True)
        print(f"[dry-run] Would upload folder -> datasets/{args.repo_id}")
        return

    if args.purge_remote and not args.prepare_only:
        purge_remote_tasks_tree(api, args.repo_id, token=args.token, dry_run=False)

    assets = prepare_staging(tasks_root, staging)
    print(f"Staged {len(assets)} task CSV(s) under {staging / disc.HF_TASKS_PREFIX}")

    if args.prepare_only:
        print("Done (--prepare-only).")
        return

    upload(staging, args.repo_id, token=args.token)


if __name__ == "__main__":
    main()
