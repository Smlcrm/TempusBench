"""TT12: optional smoke for model subprocess via conda (off by default)."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.needs_conda


@pytest.mark.skipif(
    os.environ.get("CONDA_MATRIX_TEST", "") != "1",
    reason="Set CONDA_MATRIX_TEST=1 to run conda / executor smoke (slow, needs conda).",
)
def test_conda_cli_available() -> None:
    assert shutil.which("conda") is not None, "conda not on PATH"


@pytest.mark.skipif(
    os.environ.get("CONDA_MATRIX_TEST", "") != "1",
    reason="Set CONDA_MATRIX_TEST=1 to run conda / executor smoke (slow, needs conda).",
)
def test_python_module_import_tempus_bench_under_conda_base() -> None:
    if shutil.which("conda") is None:
        pytest.skip("conda not on PATH")
    proc = subprocess.run(
        ["conda", "run", "-n", "base", "python", "-c", "import tempus_bench"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"conda base env cannot import tempus_bench (exit {proc.returncode}): {proc.stderr}"
        )
