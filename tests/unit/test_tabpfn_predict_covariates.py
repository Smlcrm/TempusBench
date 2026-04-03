"""TabPFN predict: past covariates only (no x_target at prediction time)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


def _settings() -> dict:
    root = Path(__file__).resolve().parents[2] / "tempus_bench" / "models" / "tabpfn" / "settings.yaml"
    with open(root, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_predict_rejects_x_target() -> None:
    from tempus_bench.models.tabpfn.tabpfn_model import TabpfnModel

    settings = _settings()
    m = TabpfnModel({}, settings)
    m.is_fitted = True
    y = np.zeros((5, 1), dtype=np.float64)
    ts_c = np.arange(5, dtype=np.int64)
    ts_t = np.arange(3, dtype=np.int64)
    x_c = np.zeros((5, 1), dtype=np.float64)
    x_t = np.zeros((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="x_target"):
        m.predict(y, ts_c, ts_t, x_context=x_c, x_target=x_t)
