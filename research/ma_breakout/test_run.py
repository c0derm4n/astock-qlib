"""Regression tests for research metrics that depend on holdings state."""
from __future__ import annotations

import unittest

import pandas as pd

from research.ma_breakout.run import opening_events


class OpeningEventTests(unittest.TestCase):
    def test_split_and_partial_sale_do_not_create_new_opening_event(self):
        dates = pd.bdate_range("2026-01-01", periods=5)
        holdings = pd.DataFrame([
            {"date": dates[0], "symbol": "SH515880", "quantity": 100.},
            {"date": dates[1], "symbol": "SH515880", "quantity": 200.},
            {"date": dates[2], "symbol": "SH515880", "quantity": 100.},
            {"date": dates[4], "symbol": "SH515880", "quantity": 100.},
        ])
        self.assertEqual(opening_events(holdings, dates), 2)

    def test_simultaneous_symbols_count_separate_openings(self):
        dates = pd.bdate_range("2026-01-01", periods=3)
        holdings = pd.DataFrame([
            {"date": dates[0], "symbol": "SH515880", "quantity": 100.},
            {"date": dates[0], "symbol": "SH518880", "quantity": 100.},
            {"date": dates[1], "symbol": "SH515880", "quantity": 100.},
            {"date": dates[1], "symbol": "SH518880", "quantity": 100.},
            {"date": dates[2], "symbol": "SH515880", "quantity": 100.},
        ])
        self.assertEqual(opening_events(holdings, dates), 2)


if __name__ == "__main__":
    unittest.main()
