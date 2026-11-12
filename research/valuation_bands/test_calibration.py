"""校准相关测试: Type 7 分位、年度低点、因果冻结、覆盖率与阈值顺序。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.valuation_bands import calibration as cal
from research.valuation_bands import fixtures


def build_calibration_dataset(
    start: str = "2011-01-01",
    end: str = "2016-12-31",
    low_values=(8.0, 9.0, 10.0, 11.0, 12.0),
    baseline: float = 15.0,
    drop: str | None = None,
    tie: bool = False,
):
    """构造: 每年唯一最低价日, 该日估值可控, 其余交易日估值=baseline。"""
    calendar = pd.DatetimeIndex(__import__("research.valuation_bands.data", fromlist=["x"]).build_calendar(start, end))
    n = len(calendar)
    price = np.full(n, 1000.0)
    valuation = np.full(n, baseline)
    low_dates = {}
    for offset, value in enumerate(low_values):
        year = 2011 + offset
        year_sessions = calendar[calendar.year == year]
        low_date = year_sessions[len(year_sessions) // 3]
        low_dates[year] = low_date
        price[calendar.get_loc(low_date)] = 900.0
        valuation[calendar.get_loc(low_date)] = value
        if tie:
            twin = year_sessions[len(year_sessions) // 3 + 1]
            price[calendar.get_loc(twin)] = 900.0
            valuation[calendar.get_loc(twin)] = value + 1.0
    frame = fixtures.make_dataset(
        start, end,
        price_map={"000300": price},
        valuation_map={("000300", "PE_TTM"): valuation},
        etf_map={"SH510300": {"close": np.full(n, 4.0), "open": np.full(n, 4.0)}},
    )
    if drop:
        frame = _drop(frame, drop)
    return frame, low_dates


def _drop(data, mode: str):
    valuation = data.frames["index_valuation"]
    price = data.frames["index_price"]
    if mode == "valuation_coverage":
        mask = ~(valuation["trade_date"].dt.year.eq(2013) & (np.arange(len(valuation)) % 10 == 0))
        data.frames["index_valuation"] = valuation[mask].reset_index(drop=True)
    elif mode == "price_coverage":
        data.frames["index_price"] = price.iloc[1:].reset_index(drop=True)
    elif mode == "low_day_valuation":
        target = valuation[valuation["trade_date"].dt.year.eq(2012)].iloc[0]["trade_date"]
        data.frames["index_valuation"] = valuation[valuation["trade_date"] != target].reset_index(drop=True)
    elif mode == "low_day_valuation_exact":
        price_frame = price
        year = 2012
        sessions = price_frame[price_frame["trade_date"].dt.year.eq(year)]
        low_date = sessions.loc[sessions["close"].idxmin(), "trade_date"]
        data.frames["index_valuation"] = valuation[valuation["trade_date"] != low_date].reset_index(drop=True)
    return data


class QuantileTests(unittest.TestCase):
    def test_type7_matches_numpy(self):
        np.random.seed(7)
        values = np.random.normal(10, 2, 500)
        for q in (0.05, 0.25, 0.5, 0.75, 0.85, 0.95):
            self.assertAlmostEqual(cal.quantile_type7(values, q), float(np.quantile(values, q, method="linear")), places=12)

    def test_hand_computed(self):
        # n=4, h=(n-1)q+1, q=0.85 -> 3.55 -> 3 + 0.55*(4-3) = 3.55
        self.assertAlmostEqual(cal.quantile_type7([1, 2, 3, 4], 0.85), 3.55, places=12)
        self.assertAlmostEqual(cal.quantile_type7([10, 20], 0.75), 17.5, places=12)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            cal.quantile_type7([], 0.5)


class AnnualLowTests(unittest.TestCase):
    def test_median_and_thresholds_hand_computed(self):
        data, low_dates = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000, "band_upper_mult": 1.1,
                  "band_mid_mult": 1.0, "band_low_mult": 0.9}
        snapshot = cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                                      snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertAlmostEqual(snapshot.median_low_value, 10.0, places=12)
        self.assertAlmostEqual(snapshot.l1, 11.0, places=12)
        self.assertAlmostEqual(snapshot.l2, 10.0, places=12)
        self.assertAlmostEqual(snapshot.l3, 9.0, places=12)
        expected_exit = cal.quantile_type7(data.frames["index_valuation"]["value"].tolist(), 0.85)
        self.assertAlmostEqual(snapshot.exit_level, expected_exit, places=12)
        self.assertEqual([entry["year"] for entry in snapshot.lows], [2011, 2012, 2013, 2014, 2015])
        for entry, year in zip(snapshot.lows, range(2011, 2016)):
            self.assertEqual(entry["trade_date"], str(low_dates[year].date()))
        self.assertGreater(snapshot.exit_level, snapshot.l1)

    def test_tie_uses_earliest_date(self):
        data, _ = build_calibration_dataset(tie=True)
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        snapshot = cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                                      snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        entry = snapshot.lows[0]
        self.assertEqual(entry["tie_count"], 2)
        self.assertEqual(float(entry["value"]), 8.0)  # 并列最低价取最早日期的估值

    def test_causal_no_revision_of_frozen_year(self):
        data, _ = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        stamp = pd.Timestamp("2016-01-04 19:00")
        first = cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85, snapshot_ts=stamp)
        # 修改 2016 年之后的数据 (修订/新数据), 2016 阈值必须不变
        data2, _ = build_calibration_dataset()
        valuation = data2.frames["index_valuation"].copy()
        mask = valuation["trade_date"].dt.year >= 2016
        valuation.loc[mask, "value"] = 3.0
        data2.frames["index_valuation"] = valuation
        second = cal.calibrate_year(data2, "000300", "PE_TTM", 2016, params, q=0.85, snapshot_ts=stamp)
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)
        self.assertAlmostEqual(first.exit_level, second.exit_level, places=12)

    def test_invalid_threshold_order_raises(self):
        data, _ = build_calibration_dataset(low_values=(8.0, 9.0, 10.0, 11.0, 12.0), baseline=10.5)
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertEqual(ctx.exception.reason, "invalid_threshold_order")

    def test_fixed_exit_level(self):
        data, _ = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        snapshot = cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                                      fixed_exit_level=14.34,
                                      snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertAlmostEqual(snapshot.exit_level, 14.34, places=12)

    def test_price_coverage_incomplete(self):
        data, _ = build_calibration_dataset(drop="price_coverage")
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertEqual(ctx.exception.reason, "annual_price_incomplete")

    def test_valuation_coverage_incomplete(self):
        data, _ = build_calibration_dataset(drop="valuation_coverage")
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertEqual(ctx.exception.reason, "annual_valuation_incomplete")

    def test_sample_count_insufficient(self):
        data, _ = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 10 ** 6}
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertEqual(ctx.exception.reason, "sample_count_insufficient")

    def test_low_day_valuation_missing(self):
        data, _ = build_calibration_dataset(drop="low_day_valuation_exact")
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertEqual(ctx.exception.reason, "annual_low_valuation_missing")

    def test_lookahead_availability_rejected(self):
        data, _ = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        # 快照时间早于最后一个观测的可得时间(2015-12-31 的估值在 2016-01-04 19:00 才可用)
        with self.assertRaises(cal.CalibrationError) as ctx:
            cal.calibrate_year(data, "000300", "PE_TTM", 2016, params, q=0.85,
                               snapshot_ts=pd.Timestamp("2016-01-01 19:00"))
        self.assertEqual(ctx.exception.reason, "lookahead_valuation")

    def test_book_caches_and_records_failures(self):
        data, _ = build_calibration_dataset()
        params = {"lookback_years": 5, "price_coverage_min": 1.0, "valuation_coverage_min": 0.95,
                  "min_valuation_observations": 1000}
        book = cal.CalibrationBook(data, params, "000300", "PE_TTM", "A", q=0.85)
        snapshot = book.get(2016, snapshot_ts=pd.Timestamp("2016-01-04 19:00"))
        self.assertIsNotNone(snapshot)
        self.assertIs(book.get(2016), snapshot)  # 年内固定, 不重算
        self.assertIsNone(book.get(2030, snapshot_ts=pd.Timestamp("2030-01-02 19:00")))
        self.assertIn(2030, book.failures)
        # 数据不足导致的失败允许后续周度评估重试, 成功后才冻结
        self.assertIsNone(book.get(2030, snapshot_ts=pd.Timestamp("2030-01-09 19:00")))
        self.assertIn(2030, book.failures)


if __name__ == "__main__":
    unittest.main()
