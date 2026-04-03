"""Lag-Llama: GluonTS 0.14 rejects pandas-2.2 aliases (ME, QE); normalize for PandasDataset."""

from __future__ import annotations

import pytest

from tempus_bench.utils.lagllama_freq import normalize_freq_for_lagllama


@pytest.mark.parametrize(
    ("incoming", "expected"),
    [
        ("ME", "M"),
        ("2ME", "2M"),
        ("QE", "Q"),
        ("3QE", "3Q"),
        ("YE", "Y"),
        ("YS", "AS"),
        ("M", "M"),
        ("D", "D"),
        ("W", "W"),
        ("H", "h"),
        ("T", "min"),
        ("S", "s"),
    ],
)
def test_normalize_freq_pandas2_aliases_to_gluonts_legacy(
    incoming: str, expected: str
) -> None:
    assert normalize_freq_for_lagllama(incoming) == expected


def test_normalize_freq_does_not_strip_suffix_from_arbitrary_strings() -> None:
    assert normalize_freq_for_lagllama("SOME") == "SOME"


def test_gluonts_accepts_normalized_month_freq() -> None:
    """Regression: ``ME`` raised KeyError in get_lags / time features (Batch baggage_100_multivariate)."""
    from gluonts.time_feature import get_lags_for_frequency

    assert get_lags_for_frequency(
        freq_str=normalize_freq_for_lagllama("ME"), num_default_lags=1
    )
    with pytest.raises((KeyError, ValueError)):
        get_lags_for_frequency(freq_str="ME", num_default_lags=1)
