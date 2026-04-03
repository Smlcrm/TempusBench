"""Integration test hooks: markers, collection tweaks."""

from __future__ import annotations

import pytest

_SLOW_PACKAGES = frozenset(
    {
        "sundial",
        "lstm",
        "patchtst_granite",
        "patchtst_fm",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        nid = item.nodeid.lower()
        if any(p in nid for p in _SLOW_PACKAGES):
            item.add_marker(pytest.mark.slow)
