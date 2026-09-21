"""验证趋势规则、代理边界、因果性及原现金引擎衔接。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.goal30.risk_overlay import allocated_weights, overlay_allocator, risk_policy, trend_gate, verify_trace
from research.sp500.explore import bounded_targets


class RiskOverlayTests(unittest.TestCase):
    def setUp(self):
        selected = json.loads(SELECTED.read_text(encoding='utf-8'))
        self.candidate = selected['candidate']
        self.policy = selected['policy']
        self.variant = {'name': 'test', 'trend_window': 3, 'vol_target': None}

    def test_future_prices_cannot_change_past_trend(self):
        dates = pd.bdate_range('2025-01-01', periods=12)
        prices = pd.DataFrame({'SH515880': np.arange(1., 13.)}, index=dates)
        expected = trend_gate(prices.iloc[:8], 3)
        prices.iloc[8:] = .01
        pd.testing.assert_frame_equal(expected, trend_gate(prices, 3).iloc[:8])
        prices.iloc[8:] = 1e6
        pd.testing.assert_frame_equal(expected, trend_gate(prices, 3).iloc[:8])

    def test_incomplete_equal_or_nonfinite_prices_do_not_pass(self):
        prices = pd.DataFrame({'SH515880': [1., 2., 3., np.nan, 4., 5., 5., 5., 5.]},
                              index=pd.bdate_range('2025-01-01', periods=9))
        result = trend_gate(prices, 3).iloc[:, 0]
        self.assertEqual(result.tolist(), [False, False, True, False, False, False, True, False, False])
        prices.iloc[3] = np.inf
        self.assertFalse(trend_gate(prices, 3).iloc[5, 0])
        with self.assertRaises(ValueError):
            trend_gate(prices.iloc[::-1], 3)

    def test_fallback_only_for_market_ineligibility(self):
        all_symbols = ['SH515880', 'SH518880', 'SH510300']
        gate = {s: True for s in all_symbols}
        gate['SH515880'] = False
        self.assertEqual(allocated_weights(self.candidate, all_symbols, (), gate), {'SH518880': .35})
        self.assertEqual(allocated_weights(self.candidate, ['SH510300', 'SH518880'], (), gate),
                         {'SH510300': .6, 'SH518880': .35})
        gate['SH510300'] = False
        self.assertEqual(allocated_weights(self.candidate, ['SH510300', 'SH518880'], (), gate),
                         {'SH518880': .35})
        gate['SH515880'] = True
        self.assertEqual(allocated_weights(self.candidate, all_symbols, ['SH515880'], gate),
                         {'SH518880': .35})

    def test_volatility_budget_is_active_and_does_not_add_leverage(self):
        dates = pd.bdate_range('2025-01-01', periods=60)
        returns = pd.DataFrame({'SH515880': [.1, -.1] * 30, 'SH518880': [.05, -.05] * 30}, index=dates)
        raw = self.candidate['weights']
        base = bounded_targets(raw, {}, returns, self.policy)
        policy = risk_policy(self.policy, {**self.variant, 'vol_target': .15})
        targets = bounded_targets(raw, {}, returns, policy)
        self.assertEqual(policy['volatility_policy'], 'target')
        self.assertEqual(self.policy['volatility_policy'], 'observe_only')
        self.assertLess(sum(targets.values()), sum(base.values()))
        self.assertLessEqual(sum(targets.values()), .95)

    def market(self, *, split=False):
        dates = pd.bdate_range('2025-01-01', periods=9)
        rows = []
        for symbol in ('SH515880', 'SH518880', 'SH510300'):
            values = np.arange(10., 19.)
            if symbol == 'SH515880' and not split:
                values[5:] = 8.
            factors = np.ones(9)
            if symbol == 'SH515880' and split:
                factors[5:] = 2.
            for d, price, factor in zip(dates, values, factors):
                rows.append({'date': d, 'symbol': symbol, 'open': price / factor, 'close': price / factor,
                             'volume': 1e6, 'adj_close': price})
        bars = pd.DataFrame(rows)
        market = SimpleNamespace(bars=bars, calendar=dates,
                    eligibility=bars[['date', 'symbol']].assign(eligible=True, avg_amount=1e9))
        market.wide = lambda field: bars.pivot(index='date', columns='symbol', values=field)
        return market

    def replay(self, market, actions):
        policy = {**self.policy, 'initial_capital': 100000., 'max_daily_turnover': 1.,
                  'commission': 0., 'minimum_commission': 0., 'slippage': 0., 'impact_coefficient': 0.}
        trace = []
        with patch('research.goal30.cash_replay.allocator', overlay_allocator(market.wide('adj_close'), self.variant, trace)):
            report, trades, events, holdings, _ = simulate_cash(market, self.candidate, policy,
                                                      str(market.calendar[2].date()), actions)
        audit_cash_ledger(market, report, trades, events, policy['initial_capital'])
        verify_trace(market, self.candidate, self.variant, trace)
        return report, trades, events, holdings, trace

    def test_trend_exit_executes_next_day_without_proxy_buy(self):
        market = self.market()
        report, trades, _, _, trace = self.replay(market, [])
        exit_signal = str(market.calendar[5].date())
        sales = trades[(trades.symbol == 'SH515880') & (trades.signal_date == exit_signal)]
        self.assertEqual(sales.side.tolist(), ['sell'])
        self.assertEqual(sales.date.tolist(), [str(market.calendar[6].date())])
        self.assertFalse(trades.symbol.eq('SH510300').any())
        self.assertTrue((pd.to_datetime(trades.signal_date) < pd.to_datetime(trades.date)).all())
        self.assertTrue((report.cash >= 0).all())
        # 独立验收必须拒绝把被趋势排除的资产重新加入目标。
        trace[3]['target_weights'] = json.dumps({'SH515880': .6})
        with self.assertRaises(ValueError):
            verify_trace(market, self.candidate, self.variant, trace)

    def test_split_adjustment_does_not_create_false_trend_exit(self):
        market = self.market(split=True)
        actions = [{'type': 'split', 'event_id': 'test-split', 'symbol': 'SH515880', 'ratio': 2.,
                    'effective_date': str(market.calendar[5].date())}]
        _, _, events, _, trace = self.replay(market, actions)
        event = events.iloc[0]
        self.assertEqual(event.quantity_after, event.quantity_before * 2)
        self.assertIn('SH515880', json.loads(trace[3]['raw_weights']))


if __name__ == '__main__':
    unittest.main()
