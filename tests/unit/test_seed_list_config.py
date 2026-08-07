"""Seed configuration on EvaluationConfig."""

import pytest
from pydantic import ValidationError

from tempus_bench.utils.configs import EvaluationConfig


def test_single_int_seed_becomes_a_one_element_list():
    cfg = EvaluationConfig(task_path="Synthetic Tasks/Trend", seeds=42)
    assert cfg.seed_list() == [42]


def test_list_of_seeds_is_preserved_in_order():
    cfg = EvaluationConfig(task_path="Synthetic Tasks/Trend", seeds=[42, 43, 44])
    assert cfg.seed_list() == [42, 43, 44]


def test_default_is_a_single_seed():
    cfg = EvaluationConfig(task_path="Synthetic Tasks/Trend")
    assert cfg.seed_list() == [0]


def test_empty_seed_list_is_rejected():
    with pytest.raises(ValidationError, match="at least one seed"):
        EvaluationConfig(task_path="Synthetic Tasks/Trend", seeds=[])


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        EvaluationConfig(task_path="Synthetic Tasks/Trend", seeds=[7, 7])
