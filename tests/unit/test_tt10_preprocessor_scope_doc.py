"""TT10: points to existing preprocessor / calendar suites (no duplication per model)."""

from __future__ import annotations

from pathlib import Path

from tempus_bench.utils.paths import get_project_root


def test_preprocessors_have_dedicated_unit_tests() -> None:
    root = Path(get_project_root())
    assert (root / "tests" / "unit" / "test_preprocessor_finite_output.py").is_file()
    assert (root / "tests" / "test_covariate_calendar_timestamps.py").is_file()
    assert (root / "tests" / "unit" / "test_preprocessor_normalization.py").is_file()
