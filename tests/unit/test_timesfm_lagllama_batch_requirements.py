"""Regression tests for Google Batch dependency failures (lagllama, timesfm covariates).

Batch worker logs showed:

- **lagllama**: ``ModuleNotFoundError: No module named 'gluonts.torch.modules.loss'``
  when pip resolved ``gluonts[torch]>=0.14.0,<0.16.0`` to 0.15.x (that module was removed).
- **timesfm_200m** (covariate task): ``ModuleNotFoundError: No module named 'jax'`` because
  ``timesfm.forecast_with_covariates`` imports JAX via ``timesfm.xreg_lib``.

These tests lock the *requirements contract* so the Conda env install pulls compatible deps.
"""

from __future__ import annotations

import re
from pathlib import Path

import packaging.requirements
import packaging.version
import pytest

from tempus_bench.utils.paths import get_models_dir


def _requirements_lines(req_path: Path) -> list[str]:
    return [
        stripped
        for raw in req_path.read_text(encoding="utf-8").splitlines()
        if (stripped := raw.split("#", 1)[0].strip())
    ]


def _first_line_prefix(req_path: Path, prefix: str) -> str:
    lower_prefix = prefix.lower()
    for line in _requirements_lines(req_path):
        if line.lower().startswith(lower_prefix):
            return line
    raise AssertionError(f"No line starting with {prefix!r} in {req_path}")


def test_lagllama_requirements_exclude_gluonts_015_where_loss_module_removed() -> None:
    """GluonTS 0.15+ removes ``gluonts.torch.modules.loss``; vendored lag_llama still imports it."""
    path = Path(get_models_dir()) / "lagllama" / "requirements.txt"
    line = _first_line_prefix(path, "gluonts")
    compact = line.replace(" ", "")
    assert "<0.15.0" in compact, line

    req = packaging.requirements.Requirement(line)
    bad_version = packaging.version.Version("0.15.1")
    assert not req.specifier.contains(
        bad_version, prereleases=True
    ), "Must not allow GluonTS 0.15.x (loss module removed)"
    good_version = packaging.version.Version("0.14.4")
    assert req.specifier.contains(
        good_version, prereleases=True
    ), "Must still allow GluonTS 0.14.x"


def test_lagllama_loose_upper_bound_would_allow_failing_gluonts() -> None:
    """Documents the regression: ``<0.16.0`` permits 0.15.x, which breaks lag_llama imports."""
    loose = packaging.requirements.Requirement("gluonts[torch]>=0.14.0,<0.16.0")
    assert loose.specifier.contains(
        packaging.version.Version("0.15.89"), prereleases=True
    )


@pytest.mark.parametrize("package_dir", ("timesfm_200m", "timesfm_500m"))
def test_timesfm_requirements_include_jax_for_covariate_forecast_path(package_dir: str) -> None:
    """``forecast_with_covariates`` loads ``xreg_lib``, which imports JAX."""
    path = Path(get_models_dir()) / package_dir / "requirements.txt"
    lines = _requirements_lines(path)
    joined = "\n".join(lines).lower()
    assert "jax" in joined, f"{package_dir}/requirements.txt must list jax"
    assert re.search(r"jax\[cpu\]|jaxlib", joined), (
        f"{package_dir}: expected jax[cpu] or jaxlib for explicit CPU stack in workers"
    )
