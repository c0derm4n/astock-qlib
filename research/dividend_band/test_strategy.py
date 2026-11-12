"""估值带策略的针对性单元测试（合成数据，不读取真实行情）。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.dividend_band.engine import period_mask, replay
from research.dividend_band.signals import band_weight, indicators, target_weights

TABLE = [[20.0, 0.95], [10.0, 0.60], [-999.0, 0.20]]
CANDIDATE = {'id': 'test', 'indicator': 'dividend_yield', 'table': 't', 'guard': False}
TABLES = {'t': TABLE}


def synthetic(days=1400, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2014-01-02', periods=days)
    price = 5000 * np.exp(np.cumsum(rng.normal(.0002, .011, days)))
    dividend_rate = .05 / 252
    red = np.empty(days)
    red[0] = 10000.
    for i in range(1, days):
        red[i] = red[i - 1] * (price[i] + price[i - 1] * dividend_rate) / price[i - 1]
    frame = pd.DataFrame({'red': red, 'price': price,
                          'bond': 100 * np.exp(np.cumsum(rng.normal(.00005, .002, days))),
                          'yield10': np.full(days, 2.8),
                          'bond_listed': np.ones(days, dtype=bool)}, index=index)
    return frame


class BandTests(unittest.TestCase):
    def test_band_lookup(self):
        self.assertEqual(band_weight(25., TABLE), 0.95)
        self.assertEqual(band_weight(20., TABLE), 0.95)
        self.assertEqual(band_weight(19.9, TABLE), 0.60)
        self.assertEqual(band_weight(10., TABLE), 0.60)
        self.assertEqual(band_weight(9.99, TABLE), 0.20)

    def test_period_mask(self):
        index = pd.bdate_range('2024-01-01', periods=12)
        weekly = period_mask(index, 'W')
        self.assertEqual(int(weekly.sum()), 3)
        self.assertTrue(weekly[0] and weekly[5] and weekly[10])
        monthly = period_mask(index, 'M')
        self.assertTrue(monthly[0])
        self.assertEqual(int(monthly.sum()), 1)

    def test_indicators_are_causal(self):
        frame = synthetic()
        full = indicators(frame)
        cut = 900
        partial = indicators(frame.iloc[:cut])
        for column in ('dividend_yield', 'dividend_pct', 'erp', 'erp_pct', 'ma250', 'vol60'):
            pd.testing.assert_series_equal(partial[column], full[column].iloc[:cut])
        pd.testing.assert_series_equal(partial.risk, full.risk.iloc[:cut])

    def test_implied_dividend_recovers_rate(self):
        frame = synthetic()
        signals = indicators(frame)
        observed = signals.dividend_yield.dropna()
        self.assertGreater(len(observed), 1000)
        self.assertAlmostEqual(float(observed.median()), 5.0, delta=0.2)

    def test_targets_respect_cap_and_bond_listing(self):
        frame = synthetic()
        frame.loc[frame.index < frame.index[300], 'bond_listed'] = False
        signals = indicators(frame)
        target = target_weights(frame, signals, CANDIDATE, TABLES)
        self.assertTrue(target.sum(axis=1).le(0.95 + 1e-12).all())
        self.assertTrue(target.ge(0).all().all())
        self.assertTrue(target.loc[~frame.bond_listed, 'bond'].eq(0).all())
        self.assertTrue(target.loc[~frame.bond_listed, 'red'].gt(0).any())

    def test_guard_caps_weight(self):
        frame = synthetic()
        signals = indicators(frame).copy()
        signals['risk'] = True
        guarded = target_weights(frame, signals, {**CANDIDATE, 'guard': True}, TABLES)
        self.assertTrue(guarded['red'].le(0.30 + 1e-12).all())

    def test_static_and_hold_candidates(self):
        frame = synthetic()
        signals = indicators(frame)
        hold = target_weights(frame, signals, {'id': 'h', 'indicator': 'hold'}, TABLES)
        self.assertTrue(np.allclose(hold.red, .95) and np.allclose(hold.bond, 0.))
        bond = target_weights(frame, signals, {'id': 'b', 'indicator': 'bond'}, TABLES)
        self.assertTrue(np.allclose(bond.bond, .95))
        static = target_weights(frame, signals, {'id': 's', 'indicator': 'static',
                                                 'red': .6}, TABLES)
        self.assertTrue(np.allclose(static.red, .6) and np.allclose(static.bond, .35))


class ReplayTests(unittest.TestCase):
    def prices(self, days=400):
        index = pd.bdate_range('2016-01-04', periods=days)
        rng = np.random.default_rng(3)
        return pd.DataFrame({'red': 100 * np.exp(np.cumsum(rng.normal(0, .01, days))),
                             'bond': 100 * np.exp(np.cumsum(rng.normal(0, .001, days)))},
                            index=index)

    def test_no_same_day_fill_and_self_financing(self):
        prices = self.prices()
        targets = pd.DataFrame({'red': .9, 'bond': .05}, index=prices.index)
        daily, trades = replay(prices, targets, prices.index[0].date().isoformat(), freq='M')
        self.assertTrue(len(trades) > 0)
        self.assertTrue((trades.signal_date < trades.date).all())
        error = daily.nav - (daily.prior_nav + daily.pnl - daily.trading_cost
                             - daily.replication_cost)
        self.assertLess(float(error.abs().max()), 1e-7)
        self.assertLess(abs(float(daily.cash.iloc[-1] + daily.red_value.iloc[-1]
                                  + daily.bond_value.iloc[-1] - daily.nav.iloc[-1])), 1e-7)

    def test_drift_band_holds_position(self):
        prices = self.prices()
        targets = pd.DataFrame({'red': .9, 'bond': .05}, index=prices.index)
        daily, trades = replay(prices, targets, prices.index[0].date().isoformat(), band=.30,
                               freq='M')
        self.assertEqual(len(trades), 2)  # 只有首次建仓，之后在30个百分点漂移带内不动
        self.assertLess(float(daily.cash.iloc[-1]), 0.06 * float(daily.nav.iloc[-1]))

    def test_zero_target_bypasses_band(self):
        prices = self.prices()
        targets = pd.DataFrame({'red': .9, 'bond': .05}, index=prices.index)
        targets.iloc[100:] = 0.
        daily, trades = replay(prices, targets, prices.index[0].date().isoformat(), band=.99)
        self.assertTrue((trades.side == 'sell').any())
        self.assertLess(float(daily.red_value.iloc[-1]), 1e-6)
        self.assertLess(abs(float(daily.cash.iloc[-1] - daily.nav.iloc[-1])), 1e-6)

    def test_delay_two_uses_later_price(self):
        prices = self.prices()
        targets = pd.DataFrame({'red': .9, 'bond': .05}, index=prices.index)
        _, fast = replay(prices, targets, prices.index[0].date().isoformat(), delay=1, freq='M')
        _, slow = replay(prices, targets, prices.index[0].date().isoformat(), delay=2, freq='M')
        self.assertNotEqual(fast.date.iloc[0], slow.date.iloc[0])

    def test_haircut_only_hits_red(self):
        index = pd.bdate_range('2016-01-04', periods=260)
        prices = pd.DataFrame({'red': 100., 'bond': 100.}, index=index)
        targets = pd.DataFrame({'red': .95, 'bond': 0.}, index=index)
        daily, trades = replay(prices, targets, '2016-01-04', freq='W', haircut=.006)
        self.assertEqual(len(trades), 1)  # 费用造成的小幅偏离不再触发每周最低佣金
        days = (index[-1] - index[1]).days
        red_end = float(daily.red_value.iloc[-1])
        cost = float(daily.replication_cost.sum())
        self.assertAlmostEqual(red_end, 95000. - cost, delta=1e-6)
        self.assertAlmostEqual(cost / 95000., .006 * days / 365.2425, delta=5e-4)
        self.assertEqual(float(daily.bond_value.iloc[-1]), 0.)
        self.assertAlmostEqual(float(daily.nav.iloc[-1]), red_end + 4905., delta=1e-6)

    def test_no_fill_date_postpones(self):
        prices = self.prices()
        targets = pd.DataFrame({'red': .9, 'bond': .05}, index=prices.index)
        blocked = prices.index[25]
        daily, trades = replay(prices, targets, prices.index[0].date().isoformat(),
                               band=.30, no_fill_dates=[blocked])
        self.assertNotIn(str(blocked.date()), set(trades.date))


if __name__ == '__main__':
    unittest.main()
