"""Unit tests for tempus_bench.utils.task_assets."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tempus_bench.utils.paths import get_project_root
from tempus_bench.utils.task_assets import (
    SKIP_DOWNLOAD_ENV,
    discover_expected_assets,
    ensure_task_assets,
    reset_ensure_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_ensure_cache()
    yield
    reset_ensure_cache()


def test_discover_expected_assets_finds_csvs():
    tasks_root = get_project_root() / "tempus_bench" / "tasks"
    if not tasks_root.is_dir():
        pytest.skip("tasks directory not present locally")

    specs = discover_expected_assets(tasks_root)
    assert len(specs) == 30
    assert all(spec.local_path.name.endswith(".csv") for spec in specs)
    assert all(spec.hf_path.startswith("tasks/") for spec in specs)


def test_ensure_task_assets_noop_when_complete():
    tasks_root = get_project_root() / "tempus_bench" / "tasks"
    if not tasks_root.is_dir():
        pytest.skip("tasks directory not present locally")

    missing = [
        spec
        for spec in discover_expected_assets(tasks_root)
        if not spec.local_path.is_file()
    ]
    if missing:
        pytest.skip("local task CSVs incomplete; skipping no-op test")

    with patch("tempus_bench.utils.task_assets._download_file") as mock_dl:
        ensure_task_assets()
        mock_dl.assert_not_called()


def test_ensure_task_assets_raises_when_skip_and_incomplete(tmp_path, monkeypatch):
    tasks_root = tmp_path / "tasks"
    cat = tasks_root / "univariate" / "foo_univariate"
    cat.mkdir(parents=True)
    (cat / "task.yaml").write_text(
        "task:\n  file_name: foo_univariate.csv\n  target_variable_names: [x]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SKIP_DOWNLOAD_ENV, "1")

    with patch(
        "tempus_bench.utils.task_assets._tasks_root",
        return_value=tasks_root,
    ):
        with pytest.raises(FileNotFoundError, match=SKIP_DOWNLOAD_ENV):
            ensure_task_assets()


def test_ensure_task_assets_downloads_missing_csv(tmp_path, monkeypatch):
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "univariate" / "foo_univariate"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task:\n  file_name: foo_univariate.csv\n  target_variable_names: [x]\n",
        encoding="utf-8",
    )

    downloaded: list[str] = []

    def _fake_download(repo_id, hf_path, dest):
        downloaded.append(hf_path)
        dest.write_text("col\n", encoding="utf-8")

    monkeypatch.delenv(SKIP_DOWNLOAD_ENV, raising=False)
    with patch(
        "tempus_bench.utils.task_assets._tasks_root",
        return_value=tasks_root,
    ):
        with patch(
            "tempus_bench.utils.task_assets._download_file",
            side_effect=_fake_download,
        ):
            ensure_task_assets()

    assert downloaded == ["tasks/univariate/foo_univariate/foo_univariate.csv"]
    assert (task_dir / "foo_univariate.csv").is_file()


def test_download_snapshot_uses_tempus_bench_package_root(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    package_root = project_root / "tempus_bench"
    tasks_root = package_root / "tasks"
    package_root.mkdir(parents=True)

    captured: dict[str, str] = {}

    def _fake_snapshot_download(*, repo_id, repo_type, local_dir, allow_patterns):
        captured["local_dir"] = local_dir
        # Simulate HF layout: tasks/ under local_dir
        target = Path(local_dir) / "tasks" / "univariate" / "foo"
        target.mkdir(parents=True)
        (target / "task.yaml").write_text("task:\n  file_name: foo.csv\n", encoding="utf-8")

    monkeypatch.setattr(
        "tempus_bench.utils.task_assets.get_project_root",
        lambda: project_root,
    )
    with patch(
        "huggingface_hub.snapshot_download",
        side_effect=_fake_snapshot_download,
    ):
        from tempus_bench.utils.task_assets import _download_snapshot

        _download_snapshot("Smlcrm/tempus_bench_tasks", tasks_root)

    assert captured["local_dir"] == str(package_root)
    assert tasks_root.is_dir()


def test_download_snapshot_relocates_misplaced_root_tasks(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    package_root = project_root / "tempus_bench"
    tasks_root = package_root / "tasks"
    misplaced = project_root / "tasks"
    misplaced.mkdir(parents=True)
    (misplaced / "README.md").write_text("ok", encoding="utf-8")

    def _fake_snapshot_download(*args, **kwargs):
        pass  # no-op; only test relocation path

    monkeypatch.setattr(
        "tempus_bench.utils.task_assets.get_project_root",
        lambda: project_root,
    )
    with patch(
        "huggingface_hub.snapshot_download",
        side_effect=_fake_snapshot_download,
    ):
        from tempus_bench.utils.task_assets import _download_snapshot

        _download_snapshot("Smlcrm/tempus_bench_tasks", tasks_root)

    assert tasks_root.is_dir()
    assert not misplaced.exists()
    assert (tasks_root / "README.md").is_file()


def test_get_tasks_dir_triggers_ensure(monkeypatch):
    tasks_root = get_project_root() / "tempus_bench" / "tasks"
    if not tasks_root.is_dir():
        pytest.skip("tasks directory not present locally")

    called = {"n": 0}

    def _fake_ensure():
        called["n"] += 1

    monkeypatch.setattr(
        "tempus_bench.utils.task_assets.ensure_task_assets",
        _fake_ensure,
    )
    from tempus_bench.utils import paths

    paths.get_tasks_dir()
    assert called["n"] == 1
