import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.ex_nasdaq.core import SYMBOLS, EXCLUDED, market30, validate_candidate, signals, execute
from research.sp500.explore import signal_matrix, allocator
from src.runs import read_protocol


class ExNasdaqTests(unittest.TestCase):
    def test_universe_has_exactly_original_remaining_30(self):
        self.assertEqual(len(SYMBOLS), 30)
        self.assertNotIn(EXCLUDED, SYMBOLS)
        self.assertIn('SZ159920', SYMBOLS)
        self.assertIn('SH518880', SYMBOLS)

    def test_excluded_and_external_assets_rejected_in_all_weight_locations(self):
        cases = [
            {'family': 'fixed', 'weights': {EXCLUDED: .5}},
            {'family': 'fixed', 'weights': {'USSPY': .5}},
            {'family': 'fixed', 'fallbacks': {'SH518880': {EXCLUDED: 1.}}},
            {'family': 'fixed', 'fallbacks': {EXCLUDED: {'SH518880': 1.}}},
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_candidate(candidate)

    def test_old_model_and_existing_replay_are_not_accepted(self):
        for spec in [{'family': 'existing'}, {'family': 'hybrid', 'signal': 'lightgbm'}]:
            with self.assertRaises(ValueError):
                validate_candidate(spec)

    def test_removal_covers_all_data_channels(self):
        symbols = sorted(SYMBOLS | {EXCLUDED})
        index = pd.MultiIndex.from_product([pd.to_datetime(['2020-01-02']), symbols], names=['datetime', 'instrument'])
        market = SimpleNamespace(bars=pd.DataFrame({'symbol': symbols}), eligibility=pd.DataFrame({'symbol': symbols}),
                                 features=pd.DataFrame({'x': 1.}, index=index), labels=pd.Series(1., index=index),
                                 label_ends=pd.Series(pd.Timestamp('2020-02-03'), index=index),
                                 nav=pd.DataFrame({'symbol': symbols}), meta=pd.DataFrame({'symbol': symbols}), audit={})
        with patch('research.ex_nasdaq.core.frozen_market', return_value=market):
            reduced = market30('unused')
        for name in ['bars', 'eligibility', 'nav', 'meta']:
            self.assertEqual(set(getattr(reduced, name).symbol), SYMBOLS)
        for name in ['features', 'labels', 'label_ends']:
            self.assertEqual(set(getattr(reduced, name).index.get_level_values('instrument')), SYMBOLS)

    def test_future_prices_do_not_change_scores_or_decisions(self):
        dates = pd.bdate_range('2019-01-01', periods=350)
        symbols = sorted(SYMBOLS)
        price = pd.DataFrame(np.random.default_rng(12).lognormal(0, .015, (350, 30)).cumprod(axis=0), index=dates, columns=symbols)
        changed = price.copy()
        changed.iloc[300:] *= np.linspace(.1, 10, 30)
        class Example:
            def __init__(self, p): self.p = p
            def wide(self, field): return self.p
        policy = {**read_protocol(), 'max_weight': .85, 'max_group_weight': .85,
                  'max_equity_weight': .95, 'volatility_policy': 'observe_only'}
        for signal in ['momentum', 'risk_momentum', 'regression_trend', 'pullback']:
            spec = {'family': 'rotation', 'signal': signal, 'lookbacks': [20, 60, 120], 'topk': 3, 'absolute_filter': True}
            a = signals(Example(price), spec)
            b = signals(Example(changed), spec)
            pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])
            select_a, select_b = allocator(spec, dates, dates[260]), allocator(spec, dates, dates[260])
            held_a, held_b = {}, {}
            for i in range(260, 300):
                held_a = select_a(a.iloc[i], symbols, held_a, {}, price.pct_change(fill_method=None).iloc[:i+1], policy)
                held_b = select_b(b.iloc[i], symbols, held_b, {}, changed.pct_change(fill_method=None).iloc[:i+1], policy)
                self.assertEqual(held_a, held_b)
                self.assertNotIn(EXCLUDED, held_a)

    def test_daily_trend_exit_uses_only_available_history_and_keeps_bond_parking(self):
        dates = pd.bdate_range('2020-01-01', periods=100)
        symbols = sorted(SYMBOLS)
        prices = pd.DataFrame(np.linspace(100., 80., 100)[:, None] * np.ones((1, 30)), index=dates, columns=symbols)
        class Example:
            def __init__(self, p):
                self.prices = p
                self.calendar = dates
                self.bars = pd.DataFrame({'symbol': symbols})
                self.eligibility = pd.MultiIndex.from_product([dates, symbols], names=['date','symbol']).to_frame(index=False)
                self.eligibility['eligible'] = True
            def wide(self, field): return self.prices
        candidate = {'family': 'rotation', 'signal': 'momentum', 'lookbacks': [20], 'topk': 2,
                     'absolute_filter': True, 'daily_trend_exit': 20}
        changed = prices.copy()
        changed.iloc[80:] *= 5
        captured = []
        def replay(market, *args, **kwargs):
            captured.append(market.eligibility)
            return pd.DataFrame(), pd.DataFrame(), {}
        with patch('research.ex_nasdaq.core.run_candidate', side_effect=replay):
            execute(Example(prices), candidate, {}, dates[0])
            execute(Example(changed), candidate, {}, dates[0])
        a, b = captured
        pd.testing.assert_frame_equal(a[a.date < dates[80]], b[b.date < dates[80]])
        self.assertTrue(a[a.symbol == 'SH511260'].eligible.all())
        self.assertFalse(a[a.symbol != 'SH511260'].eligible.any())


if __name__ == '__main__':
    unittest.main()
