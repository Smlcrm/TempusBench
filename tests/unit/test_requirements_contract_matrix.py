"""TT01: requirements.txt under models are non-empty and line-parseable (dependency contracts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tempus_bench.utils.paths import get_models_dir


def _req_paths() -> list[Path]:
    root = Path(get_models_dir())
    return sorted(root.glob("*/requirements.txt"))


@pytest.mark.parametrize("req_path", _req_paths(), ids=lambda p: p.parent.name)
def test_requirements_file_has_non_comment_lines(req_path: Path) -> None:
    text = req_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    substantive = [
        ln
        for ln in lines
        if ln and not ln.startswith("#")
    ]
    assert substantive, f"{req_path}: expected at least one requirement line"
