import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.runs import read_protocol
from research.sp500.benchmark import aligned_wealth
from research.sp500.explore import allocator, signal_matrix, bounded_targets


class BenchmarkTests(unittest.TestCase):
    def test_currency_conversion_uses_past_quote_and_includes_dividends(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.to_datetime(['2021-01-04', '2021-01-06'])
            for name, prices in [('spy', [100., 105.]), ('sp500_price', [100., 102.]), ('usdcny', [6., 7.])]:
                pd.DataFrame({'date': dates, 'close': prices, 'adjusted_close': prices}).to_csv(Path(tmp)/f'{name}.csv', index=False)
            wealth = aligned_wealth(tmp, pd.date_range('2021-01-04', '2021-01-06'))
            self.assertEqual(wealth.loc['2021-01-05', 'spy_total_cny'], 1.)
            self.assertAlmostEqual(wealth.loc['2021-01-06', 'spy_total_cny'], 1.05 * 7 / 6)
            self.assertAlmostEqual(wealth.loc['2021-01-06', 'sp500_price_usd'], 1.02)
            with self.assertRaises(ValueError):
                aligned_wealth(tmp, pd.date_range('2021-01-04', '2021-01-07'))

    def test_long_gaps_are_rejected_instead_of_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('spy', 'sp500_price', 'usdcny'):
                pd.DataFrame({'date': ['2021-01-01', '2021-02-01'], 'close': [100., 100.],
                              'adjusted_close': [100., 100.]}).to_csv(Path(tmp)/f'{name}.csv', index=False)
            with self.assertRaises(ValueError):
                aligned_wealth(tmp, pd.date_range('2021-01-01', '2021-02-01'))


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2021-01-04', periods=80)
        self.symbols = ['SH513100', 'SH518880', 'SH510300', 'SH511260']
        self.returns = pd.DataFrame(0., index=self.dates, columns=self.symbols)
        self.policy = {**read_protocol(), 'max_weight': .6, 'max_group_weight': .7,
                       'max_equity_weight': .95, 'vol_target': .2, 'max_daily_turnover': .3}

    def test_slow_initial_fills_continue_toward_frozen_target(self):
        callback = allocator({'family': 'fixed', 'weights': {'SH513100': .5, 'SH518880': .3}}, self.dates, self.dates[0])
        first = callback(pd.Series(dtype=float), self.symbols, {}, {}, self.returns.iloc[:1], self.policy)
        self.assertAlmostEqual(sum(first.values()), .3)
        second = callback(pd.Series(dtype=float), self.symbols, first, {}, self.returns.iloc[:2], self.policy)
        self.assertGreater(sum(second.values()), sum(first.values()))
        self.assertLessEqual(sum(second.values()), .6 + 1e-10)

    def test_eligibility_loss_exits_before_next_month(self):
        callback = allocator({'family': 'fixed', 'weights': {'SH513100': .5}}, self.dates, self.dates[0])
        first = callback(pd.Series(dtype=float), self.symbols, {}, {}, self.returns.iloc[:1], self.policy)
        next_day = callback(pd.Series(dtype=float), self.symbols[1:], first, {}, self.returns.iloc[:2], self.policy)
        self.assertNotIn('SH513100', next_day)

    def test_boundary_ties_do_not_choose_one_symbol(self):
        spec = {'family': 'rotation', 'topk': 1, 'absolute_filter': False}
        callback = allocator(spec, self.dates, self.dates[0])
        weights = callback(pd.Series({'SH513100': 1., 'SH518880': 1.}), self.symbols, {}, {}, self.returns.iloc[:1], self.policy)
        self.assertEqual(weights, {})

    def test_fallback_only_uses_currently_eligible_predeclared_asset(self):
        spec = {'family': 'fixed', 'weights': {'SH513100': .55},
                'fallbacks': {'SH513100': {'SH510300': 1.}}}
        callback = allocator(spec, self.dates, self.dates[0])
        first = callback(pd.Series(dtype=float), self.symbols[1:], {}, {}, self.returns.iloc[:1], self.policy)
        self.assertEqual(set(first), {'SH510300'})
        restored = callback(pd.Series(dtype=float), self.symbols, first, {}, self.returns.iloc[:2],
                            {**self.policy, 'max_daily_turnover': 1.})
        self.assertEqual(set(restored), {'SH513100'})
        missing = callback(pd.Series(dtype=float), ['SH518880'], restored, {}, self.returns.iloc[:3], self.policy)
        self.assertEqual(missing, {})

    def test_outside_rebalance_window_only_reduces_risk_or_exits(self):
        spec = {'family': 'fixed', 'weights': {'SH513100': .55, 'SH518880': .35},
                'execution_window': 3, 'rebalance_frequency': 'M'}
        callback = allocator(spec, self.dates, self.dates[0])
        callback(pd.Series(dtype=float), self.symbols, {}, {}, self.returns.iloc[:1], self.policy)
        current = {'SH513100': .4, 'SH518880': .2}
        held = callback(pd.Series(dtype=float), self.symbols, current, {}, self.returns.iloc[:11], self.policy)
        self.assertEqual(held, current)
        exit_target = callback(pd.Series(dtype=float), self.symbols[1:], current, {}, self.returns.iloc[:12], self.policy)
        self.assertEqual(exit_target, {'SH518880': .2})
        month_end = self.dates.get_loc(pd.Timestamp('2021-01-29'))
        rebalance = callback(pd.Series(dtype=float), self.symbols, current, {}, self.returns.iloc[:month_end+1], self.policy)
        self.assertGreater(sum(rebalance.values()), sum(current.values()))

    def test_observe_only_keeps_concentration_caps_without_volatility_timing(self):
        history = pd.DataFrame(np.random.default_rng(5).normal(0, .08, (80, 4)),
                               index=self.dates, columns=self.symbols)
        policy = {**self.policy, 'max_daily_turnover': 1.}
        raw = {'SH513100': .55, 'SH518880': .35}
        normal = bounded_targets(raw, {}, history, policy)
        observed = bounded_targets(raw, {}, history, {**policy, 'volatility_policy': 'observe_only'})
        self.assertLess(sum(normal.values()), sum(observed.values()))
        self.assertAlmostEqual(sum(observed.values()), .9)
        self.assertLessEqual(max(observed.values()), policy['max_weight'])
        oversized = bounded_targets({'SH513100': 2., 'SH518880': 2.}, {}, history,
                                    {**policy, 'volatility_policy': 'observe_only'})
        self.assertLessEqual(sum(oversized.values()), .95 + 1e-10)
        self.assertLessEqual(max(oversized.values()), policy['max_weight'])

    def test_scores_and_allocations_do_not_use_future_prices(self):
        dates = pd.bdate_range('2020-01-01', periods=300)
        prices = pd.DataFrame(np.random.default_rng(4).lognormal(0, .01, (300, 4)).cumprod(axis=0), index=dates, columns=self.symbols)
        class ExampleMarket:
            def __init__(self, frame): self.frame = frame
            def wide(self, field): return self.frame
        candidate = {'family': 'rotation', 'topk': 2, 'signal': 'risk_momentum', 'lookbacks': [20, 60, 120], 'execution_window': 3}
        old = signal_matrix(ExampleMarket(prices), candidate, None)
        changed = prices.copy()
        changed.iloc[200:] *= 3
        new = signal_matrix(ExampleMarket(changed), candidate, None)
        pd.testing.assert_frame_equal(old.iloc[:200], new.iloc[:200])
        a, b = allocator(candidate, dates, dates[150]), allocator(candidate, dates, dates[150])
        held_a, held_b = {}, {}
        for i in range(150, 200):
            returns_a = prices.iloc[:i+1].pct_change(fill_method=None)
            returns_b = changed.iloc[:i+1].pct_change(fill_method=None)
            held_a = a(old.iloc[i], self.symbols, held_a, {}, returns_a, self.policy)
            held_b = b(new.iloc[i], self.symbols, held_b, {}, returns_b, self.policy)
            self.assertEqual(held_a, held_b)


if __name__ == '__main__':
    unittest.main()
