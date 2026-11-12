"""审查回归：数据可得时间、有效覆盖率及缺行不得制造可交易状态。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.valuation_bands import data as dm, calibration as cal, fixtures
from research.valuation_bands.test_calibration import build_calibration_dataset
from research.valuation_bands import run as cli
from research.valuation_bands import fetch


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        self.data, _ = build_calibration_dataset()
        self.day = self.data.calendar[-10]

    def test_stale_bar_is_not_tradable(self):
        f = self.data.frames['etf_bars']
        self.data.frames['etf_bars'] = f[f.trade_date != self.day]
        self.assertFalse(self.data.eligibility('SH510300', self.day)['eligible'])

    def test_adv_requires_consecutive_sessions(self):
        f = self.data.frames['etf_bars']
        missing = self.data.calendar[self.data.session_pos(self.day) - 8]
        self.data.frames['etf_bars'] = f[f.trade_date != missing]
        self.assertIsNone(self.data.adv20('SH510300', self.day))

    def test_future_bar_is_not_used(self):
        f = self.data.frames['etf_bars']
        f.loc[f.trade_date == self.day, 'available_at'] = self.day + pd.Timedelta(hours=20)
        self.assertFalse(self.data.eligibility('SH510300', self.day)['eligible'])

    def test_timestamp_preserves_clock_and_normalizes_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.data.frames['etf_bars'].head(1).copy()
            f['available_at'] = '2011-01-04T12:00:00+00:00'
            p = Path(tmp) / 'bars.csv'
            f.to_csv(p, index=False)
            loaded = dm._read_csv(p, 'etf_bars')
            self.assertEqual(loaded.available_at.iloc[0], pd.Timestamp('2011-01-04 20:00'))

    def test_last_observation_is_not_available_on_same_session(self):
        dates = pd.DatetimeIndex(['2026-11-05', '2026-11-06'])
        stamps = fetch._next_session_19(dates, pd.Series([dates[-1]]))
        self.assertEqual(stamps.iloc[0], pd.Timestamp('2026-11-09 19:00'))

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.data.frames['index_valuation'].head(1)
            p = Path(tmp) / 'valuation.csv'
            pd.concat([f, f]).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, '重复主键'):
                dm._read_csv(p, 'index_valuation')

    def test_calibration_export_retains_dynamic_low_date_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'calibration.csv'
            cli.write_frame(p, pd.DataFrame([{'year': 2016, 'low_2011_date': '2011-06-01'}]), ['year'])
            self.assertIn('low_2011_date', pd.read_csv(p).columns)

    def _calibrate(self):
        return cal.calibrate_year(self.data, '000300', 'PE_TTM', 2016,
                                  {}, snapshot_ts=pd.Timestamp('2016-09-12 19:00'))

    def test_unavailable_historical_price_rejected(self):
        f = self.data.frames['index_price']
        f.loc[0, 'effective_available_at'] = pd.Timestamp('2017-01-01')
        with self.assertRaises(cal.CalibrationError) as ctx:
            self._calibrate()
        self.assertEqual(ctx.exception.reason, 'lookahead_price')

    def test_invalid_prices_cannot_count_as_full_coverage(self):
        self.data.frames['index_price'].loc[0, 'close'] = np.nan
        with self.assertRaises(cal.CalibrationError) as ctx:
            self._calibrate()
        self.assertEqual(ctx.exception.reason, 'annual_price_incomplete')

    def test_invalid_values_cannot_count_as_valuation_coverage(self):
        f = self.data.frames['index_valuation']
        f.loc[f.trade_date.dt.year.eq(2012), 'value'] = np.nan
        with self.assertRaises(cal.CalibrationError) as ctx:
            self._calibrate()
        self.assertEqual(ctx.exception.reason, 'annual_valuation_incomplete')

    def test_duplicate_observations_cannot_mask_missing_prices(self):
        f = self.data.frames['index_price']
        self.data.frames['index_price'] = pd.concat([f.iloc[1:], f.iloc[1:2]])
        with self.assertRaises(cal.CalibrationError) as ctx:
            self._calibrate()
        self.assertEqual(ctx.exception.reason, 'duplicate_observations')

    def test_optional_asset_end_does_not_shorten_main_window(self):
        bars = self.data.frames['etf_bars']
        optional = bars.head(5).copy()
        optional['symbol'] = 'SH512890'
        self.data.frames['etf_bars'] = pd.concat([bars, optional])
        case = {'strategy': 'A'}
        expected = str(bars.trade_date.max().date())
        self.assertEqual(cli.freeze_actual_end(self.data, cli.protocol(), case), expected)

    def test_one_strategy_can_freeze_when_other_has_no_valuation(self):
        self.assertIsNone(cli.freeze_actual_end(self.data, cli.protocol()))
        self.assertIsNotNone(cli.freeze_actual_end(self.data, cli.protocol(), {'strategy': 'A'}))

    def test_price_and_action_gaps_block_readiness(self):
        self.data.manifest['dataset_kind'] = 'real'
        f = self.data.frames['index_price']
        self.data.frames['index_price'] = f.iloc[1:]
        gaps = dm.check_gaps(self.data, cli.protocol())
        self.assertFalse(gaps['can_run_primary'])
        self.assertFalse(gaps['can_run_ratio_gate'])
        self.assertIn('etf_actions:SH510300', gaps['blocking_series'])


if __name__ == '__main__':
    unittest.main()
