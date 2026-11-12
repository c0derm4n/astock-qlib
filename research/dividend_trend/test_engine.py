"""交易时序、成本、自融资与趋势边界的合成测试；不把合成收益作为研究结论。"""
import unittest

import numpy as np
import pandas as pd

from .run import candidates, rebalance, replay, signal_targets


class DividendTrendTests(unittest.TestCase):
    def test_candidate_registry_is_bounded(self):
        rows = candidates()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r['id'] for r in rows}), 18)

    def test_no_signal_day_or_fill_day_return_for_new_position(self):
        dates = pd.date_range('2024-01-08', periods=4, freq='B')
        prices = pd.DataFrame({'red': [100., 120., 180., 180.], 'gold': 100.}, index=dates)
        targets = pd.DataFrame({'red': .95, 'gold': 0.}, index=dates)
        daily, trades = replay(prices, targets, str(dates[0].date()), cost=0, min_fee=0, haircut=0)
        self.assertEqual(daily.nav.iloc[0], 100000)
        self.assertEqual(daily.nav.iloc[1], 100000)
        self.assertAlmostEqual(daily.nav.iloc[2], 147500)
        self.assertEqual(trades.iloc[0]['date'], '2024-01-09')
        self.assertEqual(trades.iloc[0]['signal_date'], '2024-01-08')

    def test_exit_bypasses_band(self):
        values, cash, trades, costs = rebalance([1000., 0.], 99000., [0., 0.], .001, 5.)
        self.assertEqual(values.sum(), 0)
        self.assertEqual(cash, 99995)
        self.assertEqual(costs, 5)
        self.assertEqual(trades[0][1], 'sell')

    def test_self_financing_switch_sells_before_buys(self):
        values, cash, trades, costs = rebalance([95000., 0.], 5000., [0., .95], .001, 5.)
        self.assertEqual([t[1] for t in trades], ['sell', 'buy'])
        self.assertGreaterEqual(cash, 0)
        self.assertAlmostEqual(values.sum() + cash, 100000 - costs)

    def test_equal_ma_is_cash_and_no_future_is_used(self):
        dates = pd.bdate_range('2020-01-01', periods=330)
        prices = pd.DataFrame({'red': 100., 'gold': 100.}, index=dates)
        candidate = dict(family='rotate', ma=200, vol=.15)
        self.assertEqual(signal_targets(prices, candidate).to_numpy().sum(), 0)
        prices.iloc[-1] = [1000., 1.]
        pd.testing.assert_frame_equal(signal_targets(prices, candidate).iloc[:-1],
                                      signal_targets(prices.iloc[:-1], candidate))

    def test_vol_budget_only_reduces(self):
        dates = pd.bdate_range('2020-01-01', periods=330)
        red = 100 * np.exp(np.arange(330) * .002 + .1 * np.sin(np.arange(330)))
        prices = pd.DataFrame({'red': red, 'gold': red[::-1].copy()}, index=dates)
        limited = signal_targets(prices, dict(family='cash', ma=200, vol=.15))
        unlimited = signal_targets(prices, dict(family='cash', ma=200, vol=None))
        self.assertTrue((limited <= unlimited + 1e-12).all().all())
        self.assertTrue((limited.red < unlimited.red).any())

    def test_suspended_day_cannot_fill(self):
        dates = pd.date_range('2024-01-08', periods=4, freq='B')
        prices = pd.DataFrame(100., index=dates, columns=['red', 'gold'])
        targets = pd.DataFrame({'red': .95, 'gold': 0.}, index=dates)
        daily, trades = replay(prices, targets, '2024-01-08', haircut=0,
                                no_fill_dates=['2024-01-09'])
        self.assertEqual(trades.iloc[0]['date'], '2024-01-10')
        self.assertEqual(daily.red_value.iloc[1], 0)


if __name__ == '__main__':
    unittest.main()
