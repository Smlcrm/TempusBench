"""Unit tests for path utilities under the Tasks/ + Datasets/ layout."""

from pathlib import Path

import pytest

from tempus_bench.utils.paths import (
    find_task_directories,
    find_task_documents,
    get_dataset_path,
    get_datasets_dir,
    get_logs_path,
    get_models_dir,
    get_project_root,
    get_runs_dir,
    get_tasks_dir,
    ensure_directory_exists,
)


class TestPathFunctions:
    def test_get_project_root(self):
        expected_root = Path(__file__).parent.parent.parent
        actual_root = get_project_root()
        assert actual_root == expected_root
        assert (actual_root / "tempus_bench").exists()

    def test_get_tasks_dir(self):
        assert get_tasks_dir() == get_project_root() / "Tasks"
        assert get_tasks_dir().is_dir()

    def test_get_datasets_dir_without_ensure(self):
        assert get_datasets_dir(ensure=False) == get_project_root() / "Datasets"

    def test_get_models_dir(self):
        assert get_models_dir() == get_project_root() / "tempus_bench" / "models"

    def test_get_runs_dir(self):
        assert get_runs_dir() == get_project_root() / "runs"

    def test_get_logs_path(self):
        assert get_logs_path() == get_project_root() / "logs"

    def test_get_dataset_path(self):
        docs = find_task_documents("*")
        doc = next(iter(docs.values()))
        path = get_dataset_path(
            doc["dataset_category"],
            doc["dataset_name"],
            ensure=False,
        )
        assert path.name == f"{doc['dataset_name']}.csv"
        assert path.parent.name == doc["dataset_name"]
        assert path.parent.parent.name == doc["dataset_category"]

    def test_find_task_directories_compat(self):
        mapping = find_task_directories("*")
        assert mapping
        name, selector = next(iter(mapping.items()))
        assert "/" in selector
        assert selector.endswith(name)

    def test_ensure_directory_exists(self, tmp_path: Path):
        target = tmp_path / "a" / "b"
        ensure_directory_exists(target)
        assert target.is_dir()
