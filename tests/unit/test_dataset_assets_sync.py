"""Unit tests for Hugging Face Datasets/ sync helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tempus_bench.utils import task_assets as assets


@pytest.fixture(autouse=True)
def _reset_cache():
    assets.reset_ensure_cache()
    yield
    assets.reset_ensure_cache()


def test_ensure_fails_when_unreachable_and_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(assets, "_datasets_root", lambda: tmp_path / "Datasets")
    monkeypatch.setattr(
        assets,
        "discover_expected_datasets",
        lambda: [
            assets.DatasetAssetSpec(
                dataset_category="commerce_and_trade",
                dataset_name="Demo",
                local_path=tmp_path / "Datasets" / "commerce_and_trade" / "Demo" / "Demo.csv",
                hf_path="Datasets/commerce_and_trade/Demo/Demo.csv",
            )
        ],
    )
    monkeypatch.delenv(assets.SKIP_DATASET_SYNC_ENV, raising=False)
    monkeypatch.delenv(assets.SKIP_DOWNLOAD_ENV, raising=False)

    with patch.object(
        assets,
        "_list_remote_dataset_csv_paths",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(FileNotFoundError, match="unreachable"):
            assets.ensure_dataset_assets(force=True)


def test_ensure_downloads_missing(tmp_path: Path, monkeypatch):
    root = tmp_path / "Datasets"
    dest = root / "commerce_and_trade" / "Demo" / "Demo.csv"
    monkeypatch.setattr(assets, "_datasets_root", lambda: root)
    monkeypatch.setattr(
        assets,
        "discover_expected_datasets",
        lambda: [
            assets.DatasetAssetSpec(
                dataset_category="commerce_and_trade",
                dataset_name="Demo",
                local_path=dest,
                hf_path="Datasets/commerce_and_trade/Demo/Demo.csv",
            )
        ],
    )
    monkeypatch.delenv(assets.SKIP_DATASET_SYNC_ENV, raising=False)
    monkeypatch.delenv(assets.SKIP_DOWNLOAD_ENV, raising=False)

    def _fake_download(repo_id, revision, hf_path, dest_path: Path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("variable_name,variable_unit,timestamps,values\n", encoding="utf-8")

    with patch.object(
        assets,
        "_list_remote_dataset_csv_paths",
        return_value={"Datasets/commerce_and_trade/Demo/Demo.csv"},
    ), patch.object(assets, "_download_file", side_effect=_fake_download):
        assets.ensure_dataset_assets(force=True)

    assert dest.is_file()
