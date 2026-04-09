"""TOTO ``freq_to_seconds`` must accept pandas-2.2-style aliases from DataLoader (e.g. YE)."""

from __future__ import annotations

import unittest

from tempus_bench.models.toto.freq_seconds import freq_to_seconds

_SECONDS_PER_YEAR = 365.25 * 86400.0


class TestTotoFreqToSeconds(unittest.TestCase):
    def test_pandas_year_end_alias(self) -> None:
        got = freq_to_seconds("YE")
        self.assertAlmostEqual(got, _SECONDS_PER_YEAR, places=3)

    def test_pandas_year_end_with_multiplier(self) -> None:
        got = freq_to_seconds("2YE")
        self.assertAlmostEqual(got, 2.0 * _SECONDS_PER_YEAR, places=3)

    def test_pandas_year_start_alias(self) -> None:
        got = freq_to_seconds("YS")
        self.assertAlmostEqual(got, _SECONDS_PER_YEAR, places=3)

    def test_unknown_unit_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            freq_to_seconds("12notaunit")
        self.assertIn("Unsupported", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
