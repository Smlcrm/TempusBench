"""
Unit tests for tasks task.yaml loading via shared task_yaml_loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tempus_bench.utils.configs import EvaluationConfig
from tempus_bench.utils.config_manager import ConfigManager
from tempus_bench.utils.task_yaml_loader import (
    build_task_config,
    load_task_config_from_task_dir,
)

_FLAT_UNIVARIATE = """task:
  context_window: 50
  forecast_horizon: 24
  handle_missing: interpolate
  normalization_method: standard
  file_name: test_dataset.csv
  target_variable_names:
  - series_a
"""


class TestTaskYamlLoader:
    def test_load_flat_univariate_config(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "univariate" / "test_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(_FLAT_UNIVARIATE, encoding="utf-8")

        cfg = load_task_config_from_task_dir(task_dir)
        assert cfg.task_mode == "univariate"
        assert cfg.forecast_horizon == 24
        assert cfg.context_window == 50
        assert cfg.is_normalize() is True
        assert cfg.effective_targets() == ["series_a"]

    def test_covariate_folder_yaml(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "covariate" / "cov_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            yaml.dump(
                {
                    "task": {
                        "context_window": 8,
                        "forecast_horizon": 4,
                        "handle_missing": "interpolate",
                        "normalization_method": "none",
                        "file_name": "cov.csv",
                        "target_variable_name": "y",
                        "covariate_variable_names": ["x1"],
                    }
                }
            ),
            encoding="utf-8",
        )

        cov = build_task_config(task_dir)
        assert cov.task_mode == "covariate"
        assert cov.effective_targets() == ["y"]
        assert cov.effective_covariates() == ["x1"]

    def test_config_manager_init_tasks_uses_shared_loader(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "univariate" / "test_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(_FLAT_UNIVARIATE, encoding="utf-8")

        from unittest.mock import Mock, patch

        mgr = Mock()
        mgr.task_path = "*"
        mgr.evaluation_config = EvaluationConfig(task_path="*", max_windows=1)
        mgr.logger = Mock()

        with patch(
            "tempus_bench.utils.config_manager.find_task_directories",
            return_value={"test_task": str(task_dir)},
        ):
            result = ConfigManager.init_tasks(mgr)

        assert "test_task" in result
        assert result["test_task"].task_name == "test_task"
        assert result["test_task"].forecast_horizon == 24

    def test_empty_file_raises(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "univariate" / "empty_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="No valid task document"):
            load_task_config_from_task_dir(task_dir)

    def test_invalid_schema_raises(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "univariate" / "bad_task"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            "task:\n  forecast_horizon: 24\n",
            encoding="utf-8",
        )

        with pytest.raises(Exception):
            load_task_config_from_task_dir(task_dir)

    def test_missing_task_key_raises(self, tmp_path: Path):
        task_dir = tmp_path / "tasks" / "univariate" / "no_task_key"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text("forecast_horizon: 24\n", encoding="utf-8")

        with pytest.raises(ValueError, match="No valid task document"):
            load_task_config_from_task_dir(task_dir)
