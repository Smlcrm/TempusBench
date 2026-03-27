"""NIFTY minute tasks must not repeat Unix seconds (breaks plot time alignment)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

_OPEN_ROOT = Path(__file__).resolve().parents[2]
_NIFTY_CSV = (
    _OPEN_ROOT
    / "tempus_bench"
    / "tasks"
    / "covariate"
    / "nifty_minutes_covariate"
    / "nifty_minutes_covariate.csv"
)


def _target_timestamps(csv_path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variable_type") == "target":
                raw = json.loads(row["timestamps"])
                if isinstance(raw, list) and raw:
                    return [str(x) for x in raw]
    raise AssertionError(f"No target row in {csv_path}")


def test_nifty_minutes_target_timestamps_unique_unix() -> None:
    ts = _target_timestamps(_NIFTY_CSV)
    ux = [int(pd.Timestamp(s).timestamp()) for s in ts]
    assert len(ux) == len(set(ux)), (
        f"duplicate unix seconds in {_NIFTY_CSV.name}: "
        f"{len(ux) - len(set(ux))} duplicates"
    )
