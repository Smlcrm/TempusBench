"""Unit tests for catalog task YAML schema, discovery, and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tempus_bench.utils.paths import find_task_documents, get_tasks_dir
from tempus_bench.utils.task_yaml_loader import (
    build_task_config_from_raw,
    infer_task_mode,
)


def _raw_task(**overrides):
    base = {
        "task_name": "Demo Task",
        "task_description": "demo",
        "task_catalog": "application",
        "dataset_category": "commerce_and_trade",
        "dataset_name": "Demo_Task",
        "context_window": 4,
        "forecast_horizon": 2,
        "handle_missing": "interpolate",
        "normalization_method": "standard",
        "target_variable_names": ["y"],
        "covariate_variable_names": [],
    }
    base.update(overrides)
    return base


def test_infer_task_mode_variants():
    assert infer_task_mode(["y"], []) == "univariate"
    assert infer_task_mode(["y1", "y2"], []) == "multivariate"
    assert infer_task_mode(["y"], ["x"]) == "covariate"
    assert infer_task_mode(["y1", "y2"], ["x"]) == "covariate"


def test_build_rejects_singular_target_variable_name():
    raw = _raw_task()
    raw.pop("target_variable_names")
    raw["target_variable_name"] = "y"
    with pytest.raises(ValueError, match="target_variable_name"):
        build_task_config_from_raw(raw)


def test_build_task_config_from_raw_fields():
    tc = build_task_config_from_raw(
        _raw_task(covariate_variable_names=["x1", "x2"])
    )
    assert tc.task_mode == "covariate"
    assert tc.task_path == "commerce_and_trade/Demo Task"
    assert tc.file_name == "Demo_Task.csv"
    assert tc.dataset_category == "commerce_and_trade"
    assert tc.dataset_name == "Demo_Task"
    assert tc.task_catalog == "application"


def test_catalog_tasks_dir_exists():
    assert get_tasks_dir().name == "Tasks"
    assert get_tasks_dir().is_dir()


def test_real_catalog_has_no_singular_target_field():
    tasks_dir = get_tasks_dir()
    offenders = []
    for yaml_path in tasks_dir.rglob("*.yaml"):
        for doc in yaml.safe_load_all(yaml_path.read_text(encoding="utf-8")):
            if not doc or "task" not in doc:
                continue
            if "target_variable_name" in doc["task"]:
                offenders.append(str(yaml_path))
    assert offenders == []


def test_find_task_documents_selectors():
    all_tasks = find_task_documents("*")
    assert len(all_tasks) >= 1
    one = next(iter(all_tasks.values()))
    category = one["dataset_category"]
    name = one["task_name"]
    exact = find_task_documents(f"{category}/{name}")
    assert list(exact) == [name]
    by_cat = find_task_documents(f"{category}/*")
    assert name in by_cat
