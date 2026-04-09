"""Convert benchmark ``time_freq`` strings (pandas / DataLoader) to seconds for TOTO."""

from __future__ import annotations

import re
from typing import Union


def freq_to_seconds(freq: Union[str, float, int]) -> float:
    """
    Convert a frequency string with an increment to seconds.

    Accepts forms like:
    - '15m', '30min', '45sec', '2h', '1.5h'
    - '4w', '12mth', '1y', '250ms', '10us', '100ns'
    - pandas-style short forms: '2H', '15MIN', '30S', 'YE', 'YS', 'ME'
    - week anchors like 'W-MON' are treated as a week
    """
    if isinstance(freq, (int, float)):
        return float(freq)

    if not isinstance(freq, str) or not freq.strip():
        raise ValueError(f"Unsupported frequency: {freq!r}")

    s = freq.strip().lower().replace("µs", "us")

    if s.startswith("w-"):
        return 7 * 24 * 3600.0

    m = re.fullmatch(r"\s*(?P<val>[+-]?\d*\.?\d+)\s*(?P<unit>[a-z\-]+)\s*", s)
    if not m:
        m = re.fullmatch(r"\s*(?P<unit>[a-z\-]+)\s*", s)
        if m:
            val = 1.0
            unit = m.group("unit")
        else:
            raise ValueError(f"Could not parse frequency string: {freq!r}")
    else:
        val = float(m.group("val"))
        unit = m.group("unit")

    aliases = {
        "ns": "ns",
        "nanosecond": "ns",
        "nanoseconds": "ns",
        "us": "us",
        "microsecond": "us",
        "microseconds": "us",
        "ms": "ms",
        "millisecond": "ms",
        "milliseconds": "ms",
        "s": "s",
        "sec": "s",
        "secs": "s",
        "second": "s",
        "seconds": "s",
        "m": "min",
        "min": "min",
        "mins": "min",
        "t": "min",
        "minute": "min",
        "minutes": "min",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
        "w": "w",
        "wk": "w",
        "wks": "w",
        "week": "w",
        "weeks": "w",
        "mth": "mon",
        "mths": "mon",
        "mo": "mon",
        "mon": "mon",
        "month": "mon",
        "months": "mon",
        "me": "mon",
        # pandas 2.2+ year anchors (DataLoader uses YE for yearly spacing)
        "ye": "y",
        "ys": "y",
        # legacy pandas annual offset name
        "a": "y",
        "y": "y",
        "yr": "y",
        "yrs": "y",
        "year": "y",
        "years": "y",
        "minu": "min",
        "mins.": "min",
        "sec.": "s",
    }

    unit = aliases.get(unit, unit)

    SECS = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
        "w": 7 * 86400.0,
        "mon": 365.25 / 12 * 86400.0,
        "y": 365.25 * 86400.0,
    }

    if unit not in SECS:
        raise ValueError(f"Unsupported or non-fixed frequency unit: {unit!r} from {freq!r}")

    return float(val) * SECS[unit]
