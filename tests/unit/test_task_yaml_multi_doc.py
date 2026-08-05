"""Unit tests for catalog task YAML loading via task_yaml_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tempus_bench.utils.configs import EvaluationConfig
from tempus_bench.utils.config_manager import ConfigManager
from tempus_bench.utils.task_yaml_loader import (
    build_task_config,
    build_task_config_from_raw,
)


def _raw(**overrides):
    base = {
        "task_name": "test_task",
        "task_description": "demo",
        "task_catalog": "application",
        "dataset_category": "commerce_and_trade",
        "dataset_name": "test_task",
        "context_window": 50,
        "forecast_horizon": 24,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "target_variable_names": ["series_a"],
        "covariate_variable_names": [],
    }
    base.update(overrides)
    return base


class TestTaskYamlLoader:
    def test_load_univariate_config(self):
        cfg = build_task_config_from_raw(_raw())
        assert cfg.task_mode == "univariate"
        assert cfg.forecast_horizon == 24
        assert cfg.context_window == 50
        assert cfg.is_normalize() is True
        assert cfg.effective_targets() == ["series_a"]
        assert cfg.file_name == "test_task.csv"

    def test_covariate_config(self):
        cov = build_task_config_from_raw(
            _raw(
                target_variable_names=["y"],
                covariate_variable_names=["x1"],
                normalization_method="none",
            )
        )
        assert cov.task_mode == "covariate"
        assert cov.effective_targets() == ["y"]
        assert cov.effective_covariates() == ["x1"]

    def test_config_manager_init_tasks_uses_documents(self):
        mgr = Mock()
        mgr.task_path = "*"
        mgr.evaluation_config = EvaluationConfig(task_path="*", max_windows=1)
        mgr.logger = Mock()

        with patch("tempus_bench.utils.task_assets.ensure_dataset_assets"), patch(
            "tempus_bench.utils.paths.find_task_documents",
            return_value={"test_task": _raw()},
        ):
            result = ConfigManager.init_tasks(mgr)

        assert "test_task" in result
        assert result["test_task"].task_name == "test_task"
        assert result["test_task"].forecast_horizon == 24

    def test_legacy_folder_loader_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="no longer supported"):
            build_task_config(tmp_path / "old_task")
