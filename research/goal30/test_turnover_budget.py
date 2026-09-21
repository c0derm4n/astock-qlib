"""共享预算边界、跳空、碎股、部分退出与冻结引擎回归一致性。"""
import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.budget_cash_replay import simulate_budgeted_cash
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.goal30.turnover_budget import DailyTurnoverBudget, quote


class TurnoverBudgetTests(unittest.TestCase):
    def setUp(self):
        self.selected = json.loads(SELECTED.read_text(encoding='utf-8'))
        self.policy = {**self.selected['policy'], 'initial_capital': 100000., 'commission': .0003,
                       'minimum_commission': .2, 'max_daily_turnover': .30}

    def test_shared_budget_includes_buys_sells_and_price_gaps(self):
        budget = DailyTurnoverBudget(100000., .3)
        sell = budget.cap_quantity(2000., 'sell', 11., 1e8, self.policy, 2.)
        _, _, fill = quote(sell, 'sell', 11., 1e8, self.policy, 2.)
        budget.commit(sell, 11., fill)
        buy = budget.cap_quantity(2000., 'buy', 12., 1e8, self.policy, 2.)
        _, _, fill = quote(buy, 'buy', 12., 1e8, self.policy, 2.)
        budget.commit(buy, 12., fill)
        self.assertEqual(sell, 2000.)
        self.assertEqual(buy, 600.)
        self.assertLessEqual(budget.spent, 30000.)
        self.assertLess(sell * 11 + buy * fill, 30000.)
        remaining = budget.remaining
        # 探测/未成交不能占预算，只有commit才消耗。
        budget.cap_quantity(1000., 'buy', 10., 1e8, self.policy, 1.)
        self.assertEqual(budget.remaining, remaining)
        with self.assertRaises(ValueError):
            budget.commit(100000., 10., 10.)

    def test_lot_and_odd_lot_are_inside_same_cap(self):
        budget = DailyTurnoverBudget(10000., .3)
        self.assertEqual(budget.cap_quantity(150.5, 'sell', 10., 1e8, self.policy, 1.), 100.)
        self.assertEqual(budget.cap_quantity(50.5, 'sell', 10., 1e8, self.policy, 1., odd_exit=True), 50.5)
        self.assertEqual(budget.cap_quantity(50.5, 'buy', 10., 1e8, self.policy, 1., odd_exit=True), 0.)
        small = DailyTurnoverBudget(100., .3)
        self.assertEqual(small.cap_quantity(50.5, 'sell', 10., 1e8, self.policy, 1., odd_exit=True), 0.)
        self.assertEqual(small.spent, 0.)
        for fraction in (0., -1., np.nan, np.inf, True, 2.):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                DailyTurnoverBudget(100000., fraction)

    def test_lot_solution_is_maximal_without_crossing_budget(self):
        for side in ('buy', 'sell'):
            for price in (9.1, 10., 10.9):
                budget = DailyTurnoverBudget(100000., .3)
                quantity = budget.cap_quantity(10000., side, price, 1e7, self.policy, 3.)
                _, _, fill = quote(quantity, side, price, 1e7, self.policy, 3.)
                self.assertLessEqual(budget.charge(quantity, price, fill), budget.limit)
                _, _, next_fill = quote(quantity + 100., side, price, 1e7, self.policy, 3.)
                self.assertGreater(budget.charge(quantity + 100., price, next_fill), budget.limit)

    def market(self):
        days = pd.bdate_range('2025-01-01', periods=15)
        rows = []
        for symbol in ('SH510300', 'SH515880', 'SH518880'):
            for i, day in enumerate(days):
                # 真实执行价有小幅跳空，成交仍落在原价格限制阈值内。
                opened = 10. if i % 2 == 0 else 10.4
                rows.append({'date': day, 'symbol': symbol, 'open': opened, 'close': 10.,
                             'adj_close': 10., 'volume': 1e6})
        bars = pd.DataFrame(rows)
        result = SimpleNamespace(calendar=days, bars=bars,
                                 eligibility=bars[['date', 'symbol']].assign(eligible=True, avg_amount=1e8))
        result.wide = lambda field: result.bars.pivot(index='date', columns='symbol', values=field)
        return result

    def test_disabled_extension_matches_frozen_engine(self):
        market = self.market()
        original = simulate_cash(market, self.selected['candidate'], self.policy, '2025-01-02', [])
        extended = simulate_budgeted_cash(market, self.selected['candidate'], self.policy, '2025-01-02', [], enforce_turnover=False)
        for old, new in zip(original[:4], extended[:4]):
            pd.testing.assert_frame_equal(old, new, check_exact=True)
        self.assertEqual(original[4], {k: v for k, v in extended[4].items() if k != 'turnover_clipped'})

    def test_full_exit_is_spread_across_days_and_does_not_reenter(self):
        market = self.market()
        threshold = market.calendar[5]
        def factory(candidate, calendar, start):
            def select(scores, eligible, current, holding_days, returns, policy, **kwargs):
                return {'SH515880': .85} if returns.index[-1] < threshold else {}
            return select
        result = simulate_budgeted_cash(market, self.selected['candidate'], self.policy,
                    '2025-01-02', [], select_factory=factory)
        report, trades, actions, holdings, issues = result
        self.assertTrue((report.turnover <= .3 + 1e-12).all())
        self.assertTrue((report.execution_turnover <= .3 + 1e-12).all())
        self.assertTrue((report.budget_turnover <= .3 + 1e-12).all())
        sales = trades[trades.side.eq('sell')]
        self.assertGreaterEqual(sales.date.nunique(), 3)
        self.assertFalse(holdings.date.eq(report.index[-1]).any())
        self.assertGreater(issues['turnover_clipped'], 0)
        self.assertTrue((trades.turnover_budget_after >= 0).all())
        audit_cash_ledger(market, report, trades, actions, self.policy['initial_capital'])

    def test_suspension_consumes_no_budget_and_odd_exit_finishes(self):
        market = self.market()
        # 第五日按1.015折算，造出非整手余量，验证最终清空不被整手容量截断。
        day = market.calendar[5]
        after = market.bars.symbol.eq('SH515880') & market.bars.date.ge(day)
        market.bars.loc[after, ['open', 'close']] /= 1.015
        market.bars.loc[market.bars.date.eq(market.calendar[7]), 'volume'] = 0.
        def factory(candidate, calendar, start):
            return lambda scores, eligible, current, holding_days, returns, policy, **kw: (
                {'SH515880': .85} if returns.index[-1] < market.calendar[6] else {})
        actions = [{'event_id': 'odd-split', 'symbol': 'SH515880', 'type': 'split', 'ratio': 1.015,
                    'effective_date': str(day.date())}]
        report, trades, events, holdings, _ = simulate_budgeted_cash(market, self.selected['candidate'], self.policy,
                                                   '2025-01-02', actions, select_factory=factory)
        suspended = report.loc[market.calendar[7]]
        self.assertEqual(suspended.budget_turnover, 0.)
        self.assertFalse(trades.date.eq(str(market.calendar[7].date())).any())
        self.assertTrue((report.execution_turnover <= .3 + 1e-12).all())
        self.assertTrue((trades[trades.side.eq('buy')].quantity % 100 == 0).all())
        self.assertFalse(holdings.date.eq(report.index[-1]).any())
        self.assertTrue(((trades[trades.side.eq('sell')].quantity % 100) != 0).any())
        audit_cash_ledger(market, report, trades, events, self.policy['initial_capital'])


if __name__ == '__main__':
    unittest.main()
