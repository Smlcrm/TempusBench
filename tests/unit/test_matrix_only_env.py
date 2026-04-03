"""MODEL_MATRIX_ONLY filtering for sorted_model_packages()."""

from __future__ import annotations

import pytest

from tests.integration import model_matrix_support as mms


def test_sorted_model_packages_matrix_only_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mms, "get_available_models", lambda: ["zebra", "alpha", "beta"])
    monkeypatch.delenv("MODEL_MATRIX_ONLY", raising=False)
    assert mms.sorted_model_packages() == ["alpha", "beta", "zebra"]
    monkeypatch.setenv("MODEL_MATRIX_ONLY", "beta,zebra")
    assert mms.sorted_model_packages() == ["beta", "zebra"]


def test_sorted_model_packages_matrix_only_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mms, "get_available_models", lambda: ["a"])
    monkeypatch.setenv("MODEL_MATRIX_ONLY", "nope")
    with pytest.raises(ValueError, match="unknown package id"):
        mms.sorted_model_packages()


def test_matrix_only_empty_parsed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mms, "get_available_models", lambda: ["a"])
    monkeypatch.setenv("MODEL_MATRIX_ONLY", "  ,  ")
    with pytest.raises(ValueError, match="no package ids"):
        mms.sorted_model_packages()
