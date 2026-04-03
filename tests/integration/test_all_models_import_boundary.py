"""TT02: layout + full import for every model package.

Fast checks run always. Full ``find_model_class`` imports are ``slow`` (TensorFlow / JAX, etc.).

Use ``MODEL_MATRIX_ONLY=<package_id>`` to parametrize layout/import tests for a single model."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tempus_bench.utils.paths import get_models_dir

from tests.integration.model_matrix_support import find_model_class, sorted_model_packages


@pytest.mark.parametrize("package", sorted_model_packages())
def test_model_package_layout_settings_and_impl(package: str) -> None:
    root = Path(get_models_dir()) / package
    impl = root / f"{package}_model.py"
    settings_path = root / "settings.yaml"
    assert impl.is_file(), f"missing {impl}"
    assert settings_path.is_file(), f"missing {settings_path}"
    with open(settings_path, encoding="utf-8") as f:
        yaml.safe_load(f)


@pytest.mark.slow
@pytest.mark.parametrize("package", sorted_model_packages())
def test_find_model_class_resolves(package: str) -> None:
    cls = find_model_class(package)
    assert isinstance(cls, type)
