"""Regression: ThetaHyperparams must be usable with Pydantic v2.12+ and PEP 563."""

from __future__ import annotations

import pytest

from tempus_bench.models.theta.theta_model import (
    ThetaEstimationMethod,
    ThetaHyperparams,
)


def test_theta_hyperparams_creates_with_strenum_field() -> None:
    h = ThetaHyperparams(sp=12, theta_method="correlation_optimal")
    assert h.sp == 12
    assert h.theta_method == ThetaEstimationMethod.correlation_optimal


def test_theta_hyperparams_least_squares() -> None:
    h = ThetaHyperparams(sp=4, theta_method="least_squares", use_reduced_rank=True)
    assert h.theta_method == ThetaEstimationMethod.least_squares
    assert h.use_reduced_rank is True


def test_theta_hyperparams_rejects_invalid_method() -> None:
    with pytest.raises(Exception):
        ThetaHyperparams(sp=12, theta_method="invalid_method")


def test_theta_hyperparams_rejects_zero_sp() -> None:
    with pytest.raises(Exception):
        ThetaHyperparams(sp=0, theta_method="correlation_optimal")
