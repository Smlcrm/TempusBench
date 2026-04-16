"""Hyperparameter tuner tqdm must not write to stderr on non-TTY stdout (Batch → ERROR severity)."""

from __future__ import annotations

import os
import subprocess
import sys


def _run_snippet(env: dict | None = None) -> subprocess.CompletedProcess[str]:
    """Mirror the tqdm branch in hyperparameter_tuner.optimize_hyperparameters."""
    code = """
import os
import sys
from tqdm import tqdm

_force_tqdm = os.environ.get("TEMPUSBENCH_FORCE_TQDM", "").strip() == "1"
_show_hp_bar = _force_tqdm or sys.stdout.isatty()
for _ in tqdm(
    range(3),
    desc="Hyperparameter Combinations",
    file=sys.stdout,
    mininterval=0.0,
    disable=not _show_hp_bar,
):
    pass
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
    )


def test_hyperparameter_tqdm_no_stdout_no_stderr_when_not_tty():
    """Piped stdout → not a TTY → bar disabled → no tqdm noise on either stream."""
    proc = _run_snippet()
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "Hyperparameter Combinations" not in proc.stderr
    assert "Hyperparameter Combinations" not in proc.stdout


def test_hyperparameter_tqdm_force_env_writes_stdout_not_stderr():
    """TEMPUSBENCH_FORCE_TQDM=1 on non-TTY still sends bar to stdout (mirrors file=sys.stdout)."""
    proc = _run_snippet(env={"TEMPUSBENCH_FORCE_TQDM": "1"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "Hyperparameter Combinations" in proc.stdout
    assert "Hyperparameter Combinations" not in proc.stderr
