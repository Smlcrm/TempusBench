"""Frequency strings for Lag-Llama / GluonTS (no torch imports — safe for lightweight tests)."""

from __future__ import annotations

import re

# Single-letter legacy codes GluonTS 0.14 expects; map hour/min/sec to pandas spellings.
_FREQ_LEGACY_SUBDAY_TO_PANDAS = {
    "H": "h",
    "T": "min",
    "S": "s",
}


def normalize_freq_for_lagllama(freq: str) -> str:
    """Map pipeline / pandas-2.2 freq aliases to strings GluonTS + pandas accept.

    Multivariate tasks often carry ``ME`` / ``QE`` / ``YE`` from metadata; GluonTS 0.14
    raises ``KeyError`` / invalid frequency for those aliases. Normalize to legacy
    ``M`` / ``Q`` / ``Y`` / ``AS`` first, then hour/minute/second spellings for pandas.
    """
    s = str(freq).strip()
    s = re.sub(r"^(\d*)ME$", r"\1M", s)
    s = re.sub(r"^(\d*)QE$", r"\1Q", s)
    s = re.sub(r"^(\d*)YE$", r"\1Y", s)
    s = re.sub(r"^(\d*)YS$", r"\1AS", s)
    return _FREQ_LEGACY_SUBDAY_TO_PANDAS.get(s, s)
