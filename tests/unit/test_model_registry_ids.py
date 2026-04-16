"""TT11: every discovered model id is accepted by ModelConfig."""

from __future__ import annotations

import pytest

from tempus_bench.utils.configs import ModelConfig
from tempus_bench.utils.paths import get_available_models


@pytest.mark.parametrize("model_id", sorted(get_available_models()))
def test_model_config_accepts_discovered_id(model_id: str) -> None:
    cfg = ModelConfig(model_name=model_id)
    assert cfg.model_name == model_id
