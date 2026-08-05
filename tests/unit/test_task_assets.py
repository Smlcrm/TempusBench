"""Unit tests for tempus_bench.utils.task_assets (Datasets sync)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tempus_bench.utils.paths import get_project_root
from tempus_bench.utils.task_assets import (
    SKIP_DOWNLOAD_ENV,
    SKIP_DATASET_SYNC_ENV,
    DatasetAssetSpec,
    discover_expected_datasets,
    ensure_dataset_assets,
    reset_ensure_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_ensure_cache()
    yield
    reset_ensure_cache()


def test_discover_expected_datasets_from_tasks():
    tasks_root = get_project_root() / "Tasks"
    if not tasks_root.is_dir():
        pytest.skip("Tasks directory not present locally")

    specs = discover_expected_datasets()
    assert len(specs) >= 1
    assert all(spec.local_path.name.endswith(".csv") for spec in specs)
    assert all(spec.hf_path.startswith("Datasets/") for spec in specs)


def test_ensure_dataset_assets_noop_when_complete():
    tasks_root = get_project_root() / "Tasks"
    datasets_root = get_project_root() / "Datasets"
    if not tasks_root.is_dir() or not datasets_root.is_dir():
        pytest.skip("Tasks/Datasets not present locally")

    missing = [
        spec for spec in discover_expected_datasets() if not spec.local_path.is_file()
    ]
    if missing:
        pytest.skip("local dataset CSVs incomplete; skipping no-op test")

    with patch("tempus_bench.utils.task_assets._download_file") as mock_dl:
        ensure_dataset_assets()
        mock_dl.assert_not_called()


def test_ensure_raises_when_skip_and_incomplete(tmp_path, monkeypatch):
    dest = tmp_path / "Datasets" / "commerce_and_trade" / "Demo" / "Demo.csv"
    monkeypatch.setattr(
        "tempus_bench.utils.task_assets._datasets_root",
        lambda: tmp_path / "Datasets",
    )
    monkeypatch.setattr(
        "tempus_bench.utils.task_assets.discover_expected_datasets",
        lambda: [
            DatasetAssetSpec(
                dataset_category="commerce_and_trade",
                dataset_name="Demo",
                local_path=dest,
                hf_path="Datasets/commerce_and_trade/Demo/Demo.csv",
            )
        ],
    )
    monkeypatch.setenv(SKIP_DATASET_SYNC_ENV, "1")

    with pytest.raises(FileNotFoundError, match=SKIP_DATASET_SYNC_ENV):
        ensure_dataset_assets(force=True)


def test_get_datasets_dir_triggers_ensure(monkeypatch):
    called = {"n": 0}

    def _fake_ensure():
        called["n"] += 1

    monkeypatch.setattr(
        "tempus_bench.utils.task_assets.ensure_dataset_assets",
        _fake_ensure,
    )
    from tempus_bench.utils import paths

    paths.get_datasets_dir(ensure=True)
    assert called["n"] == 1
