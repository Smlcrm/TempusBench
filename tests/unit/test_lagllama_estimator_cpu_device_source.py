"""Regression: GluonTS predictor must not force CUDA when the estimator is on CPU."""

from __future__ import annotations

from pathlib import Path


def test_lag_llama_estimator_create_predictor_uses_self_device() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "tempus_bench" / "models" / "lagllama" / "lag_llama" / "gluon" / "estimator.py"
    text = path.read_text(encoding="utf-8")
    start = text.find("    def create_predictor(")
    assert start != -1
    block = text[start:]
    assert block.count("device=self.device") == 2
    assert "torch.cuda" not in block
