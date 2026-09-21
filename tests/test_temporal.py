import unittest
import pandas as pd
from src.temporal import label_end_dates, purged_period
from src.universe_filter import historical_universe, eligibility_intervals


class TemporalTests(unittest.TestCase):
    def test_label_boundary_uses_trading_days(self):
        cal = pd.bdate_range("2025-11-01", "2026-03-01")
        period = purged_period(("2025-11-01", "2025-12-31"), cal, 20, "2026-01-01")
        ends = label_end_dates(cal, 20)
        self.assertLess(ends.loc[period[1]], pd.Timestamp("2026-01-01"))
        next_pos = cal.get_loc(period[1]) + 1
        self.assertGreaterEqual(ends.loc[cal[next_pos]], pd.Timestamp("2026-01-01"))

    def test_announcements_and_liquidity_do_not_backfill(self):
        dates = pd.bdate_range("2025-01-01", periods=65)
        bars = pd.DataFrame({"date": dates, "symbol": "SH510300", "close": 4.,
                             "amount": [100.] * 40 + [1.] * 25})
        nav = pd.DataFrame({"symbol": ["SH510300"], "nav_date": [dates[0]],
                            "ann_date": [dates[25]], "net_asset": [100.]})
        meta = pd.DataFrame(columns=["symbol", "list_date"])
        frame = historical_universe(bars, nav, meta, min_days=20, min_amount=50, min_aum=50)
        self.assertFalse(frame.loc[frame["date"] <= dates[25], "eligible"].any())
        self.assertTrue(frame.loc[frame["date"] == dates[26], "eligible"].item())
        self.assertFalse(frame["eligible"].iloc[-1])
        truncated = historical_universe(bars.iloc[:40], nav, meta, min_days=20, min_amount=50, min_aum=50)
        pd.testing.assert_series_equal(frame["eligible"].iloc[:40], truncated["eligible"])
        self.assertEqual(len(eligibility_intervals(frame)["SH510300"]), 1)

    def test_missing_aum_is_not_treated_as_eligible(self):
        dates = pd.bdate_range("2025-01-01", periods=30)
        bars = pd.DataFrame({"date": dates, "symbol": "SH510300", "close": 4., "amount": 100.})
        result = historical_universe(bars, pd.DataFrame(), pd.DataFrame(), min_days=20, min_amount=1)
        self.assertFalse(result["eligible"].any())


if __name__ == "__main__":
    unittest.main()
