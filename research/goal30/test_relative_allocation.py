"""动态配置的规则、价格前缀因果性及真实现金执行衔接测试。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.goal30.relative_allocation import (PROTOCOL, allocation_factory, feature_scores,
                                               prefix_scores, raw_allocation, verify_trace)


class RelativeAllocationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        self.spec.update(lookbacks=[2, 3, 4], volatility_window=2, start='2025-01-07', end='2025-01-16')
        self.selected = json.loads(SELECTED.read_text(encoding='utf-8'))
        self.symbols = self.spec['signal_symbols']

    def prices(self):
        return pd.DataFrame({self.symbols[0]: np.linspace(10, 11, 12),
                             self.symbols[1]: np.linspace(20, 24, 12),
                             self.symbols[2]: np.linspace(8, 10, 12)},
                            index=pd.bdate_range('2025-01-01', periods=12))

    def test_future_mutation_and_price_scale_invariance(self):
        prices = self.prices()
        expected = feature_scores(prices.iloc[:8], self.spec)
        prices.iloc[8:] = 1e6
        changed = feature_scores(prices, self.spec)
        scaled = feature_scores(prices * [4., 7., .3], self.spec)
        for n in self.spec['lookbacks']:
            pd.testing.assert_frame_equal(expected[n], changed[n].iloc[:8])
            np.testing.assert_allclose(scaled[n], changed[n], rtol=1e-10, atol=1e-10)
        direct = prefix_scores(prices.iloc[:8], self.spec)
        for n in self.spec['lookbacks']:
            np.testing.assert_allclose(direct.loc[n], expected[n].iloc[-1], rtol=1e-10, atol=1e-10)

    def test_history_gaps_flat_prices_and_invalid_spec(self):
        prices = self.prices()
        for n, frame in feature_scores(prices, self.spec).items():
            self.assertTrue(frame.iloc[:4].isna().all().all())
        for invalid in (np.nan, np.inf, 0., -1.):
            bad = prices.copy()
            bad.iloc[5, 1] = invalid
            features = feature_scores(bad, self.spec)
            self.assertTrue(all(frame.iloc[5:10, 1].isna().all() for frame in features.values()))
        flat = feature_scores(prices * 0 + 10., self.spec)
        self.assertTrue(all(frame.iloc[4:].eq(0).all().all() for frame in flat.values()))
        with self.assertRaises(ValueError):
            feature_scores(prices, {**self.spec, 'signal_symbols': ['SH513100']})
        with self.assertRaises(ValueError):
            feature_scores(prices.iloc[::-1], self.spec)

    def test_proportional_equal_and_exit_scope(self):
        matrix = pd.DataFrame([[1., 2., -3.]] * 3, index=[2, 3, 4], columns=self.symbols)
        raw = raw_allocation(matrix, 'proportional', self.symbols, (), .9)
        np.testing.assert_allclose(list(raw.values()), [.3, .6])
        self.assertEqual(raw_allocation(matrix, 'equal', self.symbols, (), .9),
                         {self.symbols[0]: .45, self.symbols[1]: .45})
        self.assertEqual(raw_allocation(matrix, 'proportional', self.symbols, [self.symbols[1]], .9),
                         {self.symbols[0]: .9})
        self.assertEqual(raw_allocation(matrix, 'equal', [], (), .9), {})
        self.assertEqual(raw_allocation(-matrix.abs(), 'proportional', self.symbols, (), .9), {})

    def test_votes_split_ties_and_leave_uninvested_sleeves_in_cash(self):
        matrix = pd.DataFrame([[2., 2., 1.], [-1., -1., -1.], [1., 2., 3.]],
                              index=[2, 3, 4], columns=self.symbols)
        raw = raw_allocation(matrix, 'votes', self.symbols, (), .9)
        np.testing.assert_allclose([raw[s] for s in self.symbols], [.15, .15, .3])
        self.assertAlmostEqual(sum(raw.values()), .6)
        matrix.iloc[0, 1] = np.nan
        self.assertNotIn(self.symbols[1], raw_allocation(matrix, 'votes', self.symbols, (), .9))

    def test_cash_engine_next_day_split_and_dividend_integration(self):
        prices = self.prices()
        bars = []
        for symbol in self.symbols:
            for i, day in enumerate(prices.index):
                value = prices.at[day, symbol]
                raw = value / (2 if symbol == 'SH515880' and i >= 7 else 1)
                if symbol == 'SH518880' and i >= 8:
                    raw -= .1
                bars.append({'date': day, 'symbol': symbol, 'adj_close': value, 'open': raw,
                             'close': raw, 'volume': 1e6})
        bars = pd.DataFrame(bars)
        market = SimpleNamespace(calendar=prices.index, bars=bars,
                                 eligibility=bars[['date', 'symbol']].assign(eligible=True, avg_amount=1e9))
        market.wide = lambda field: bars.pivot(index='date', columns='symbol', values=field)
        actions = [{'event_id': 'split-test', 'type': 'split', 'symbol': 'SH515880', 'ratio': 2.,
                    'effective_date': str(prices.index[7].date())},
                   {'event_id': 'dividend-test', 'type': 'cash_dividend', 'symbol': 'SH518880',
                    'record_date': str(prices.index[7].date()), 'ex_date': str(prices.index[8].date()),
                    'payment_date': str(prices.index[10].date()), 'cash_per_share': .1}]
        policy = {**self.selected['policy'], 'initial_capital': 100000., 'max_daily_turnover': 1.,
                  'commission': 0., 'slippage': 0., 'impact_coefficient': 0., 'minimum_commission': 0.}
        variant = {'method': 'equal', 'name': 'test'}
        trace = []
        with patch('research.goal30.cash_replay.allocator', allocation_factory(prices, self.spec, variant, trace)):
            report, trades, events, holdings, _ = simulate_cash(market, self.selected['candidate'], policy,
                                                               self.spec['start'], actions)
        self.assertTrue(audit_cash_ledger(market, report, trades, events, policy['initial_capital'])['max_nav_error'] < 1e-6)
        self.assertTrue(all(prices.index.get_loc(pd.Timestamp(r.date)) - prices.index.get_loc(pd.Timestamp(r.signal_date)) == 1
                            for r in trades.itertuples()))
        self.assertEqual(verify_trace(market, self.spec, variant, policy, trace)['signals_verified'], len(report) - 1)
        # 冻结行情为微秒，JSON/CSV解析可为纳秒；同一日期不应误判为不同交易日。
        market.calendar = market.calendar.as_unit('us')
        self.assertEqual(verify_trace(market, self.spec, variant, policy, trace)['signals_verified'], len(report) - 1)
        self.assertGreater(events.loc[events.type.eq('record_entitlement'), 'quantity'].iloc[0], 0)
        self.assertGreater(report.dividend_receivable.max(), 0)
        self.assertEqual(report.dividend_receivable.iloc[-1], 0)
        split = events[events.type.eq('split')].iloc[0]
        self.assertGreater(split.quantity_before, 0)
        self.assertEqual(split.quantity_after, split.quantity_before * 2)
        trace[-1]['raw_weights'] = json.dumps({'SH513100': .9})
        with self.assertRaises(ValueError):
            verify_trace(market, self.spec, variant, policy, trace)


if __name__ == '__main__':
    unittest.main()
