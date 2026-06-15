"""
Unit tests for tasks flat task.yaml schema, discovery, and loader validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import EvaluationConfig
from tempus_bench.utils.paths import find_task_directories, get_tasks_dir
from tempus_bench.utils.task_yaml_loader import (
    build_task_config,
    build_task_configs,
    load_task_config_from_task_dir,
)


def _write_univariate_task(task_dir: Path, *, file_name: str = "task.csv") -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        yaml.dump(
            {
                "task": {
                    "context_window": 4,
                    "forecast_horizon": 2,
                    "handle_missing": "interpolate",
                    "normalization_method": "standard",
                    "file_name": file_name,
                    "target_variable_names": ["series_a"],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_multivariate_task(task_dir: Path, *, file_name: str = "task.csv") -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        yaml.dump(
            {
                "task": {
                    "context_window": 4,
                    "forecast_horizon": 2,
                    "handle_missing": "interpolate",
                    "normalization_method": "none",
                    "file_name": file_name,
                    "target_variable_names": ["y", "x1", "x2"],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_covariate_task(task_dir: Path, *, file_name: str = "task.csv") -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        yaml.dump(
            {
                "task": {
                    "context_window": 4,
                    "forecast_horizon": 2,
                    "handle_missing": "interpolate",
                    "normalization_method": "none",
                    "file_name": file_name,
                    "target_variable_name": "y",
                    "covariate_variable_names": ["x1", "x2"],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_tasks_csv(task_dir: Path, file_name: str, rows: list[dict]) -> None:
    import csv

    path = task_dir / file_name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variable_name",
                "variable_unit",
                "timestamps",
                "values",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class TestFlatTaskYamlSchema:
    def test_univariate_config_fields(self, tmp_path: Path):
        root = tmp_path / "tasks" / "univariate" / "foo_task"
        _write_univariate_task(root)
        cfg = load_task_config_from_task_dir(root)
        assert cfg.task_mode == "univariate"
        assert cfg.effective_targets() == ["series_a"]
        assert cfg.effective_covariates() == []
        assert cfg.is_normalize() is True
        assert cfg.dataset.normalize is True
        assert cfg.dataset.file_name == "task.csv"

    def test_multivariate_and_covariate_folders(self, tmp_path: Path):
        multi_root = tmp_path / "tasks" / "multivariate" / "bar_task"
        cov_root = tmp_path / "tasks" / "covariate" / "bar_task_cov"
        _write_multivariate_task(multi_root)
        _write_covariate_task(cov_root)
        joint = load_task_config_from_task_dir(multi_root)
        cov = load_task_config_from_task_dir(cov_root)
        assert joint.task_mode == "multivariate"
        assert joint.effective_targets() == ["y", "x1", "x2"]
        assert joint.effective_covariates() == []
        assert cov.task_mode == "covariate"
        assert cov.effective_targets() == ["y"]
        assert cov.effective_covariates() == ["x1", "x2"]
        assert cov.task_path != joint.task_path

    def test_normalization_method_none(self, tmp_path: Path):
        root = tmp_path / "tasks" / "univariate" / "no_norm"
        _write_univariate_task(root, file_name="no_norm.csv")
        yaml_path = root / "task.yaml"
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        data["task"]["normalization_method"] = "none"
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        cfg = load_task_config_from_task_dir(root)
        assert cfg.is_normalize() is False
        assert cfg.dataset.normalize is False

    def test_find_task_directories_three_folders(self, tmp_path: Path, monkeypatch):
        tasks_root = tmp_path / "tasks"
        multi = tasks_root / "multivariate" / "mv_task"
        cov = tasks_root / "covariate" / "cov_task"
        _write_multivariate_task(multi)
        _write_covariate_task(cov)
        monkeypatch.setattr(
            "tempus_bench.utils.paths.get_tasks_dir",
            lambda: tasks_root,
        )
        all_tasks = find_task_directories("*")
        assert set(all_tasks) == {"mv_task", "cov_task"}
        only_cov = find_task_directories("covariate/cov_task")
        assert list(only_cov) == ["cov_task"]

    def test_loader_missing_variable_name_raises(self, tmp_path: Path):
        root = tmp_path / "tasks" / "univariate" / "missing_row"
        _write_univariate_task(root)
        _write_tasks_csv(
            root,
            "task.csv",
            [
                {
                    "variable_name": "other_name",
                    "variable_unit": "unit",
                    "timestamps": '["2020-01-01T00:00:00Z"]',
                    "values": "[1.0]",
                }
            ],
        )
        cfg = load_task_config_from_task_dir(root)
        eval_cfg = EvaluationConfig(task_path="*", max_windows=1)
        with pytest.raises(ValueError, match="missing target variable_name"):
            DataLoader(cfg, eval_cfg)


class TestCatalogTasks2Yaml:
    @pytest.mark.parametrize(
        "task_name,path",
        sorted(find_task_directories("*").items()),
        ids=sorted(find_task_directories("*").keys()),
    )
    def test_catalog_task_yaml_builds(self, task_name: str, path: str):
        configs = build_task_configs(task_name, Path(path))
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.task_name == task_name
        assert cfg.file_name
        assert cfg.effective_targets()

    def test_tasks_dir_points_at_tasks(self):
        assert get_tasks_dir().name == "tasks"

    def test_catalog_has_thirty_tasks(self):
        assert len(find_task_directories("*")) == 30
