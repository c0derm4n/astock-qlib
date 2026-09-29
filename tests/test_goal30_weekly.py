"""周度触发式调仓与双档目标判定的离线用例。"""
import unittest

import pandas as pd

from research.goal30.weekly_goal import cadence_audit, goal_checks, weekly_overlay_allocator

CANDIDATE = {
    'name': 'weekly_test_fixed',
    'family': 'fixed',
    'weights': {'SH515880': 0.6, 'SH518880': 0.35},
    'fallbacks': {'SH515880': {'SH510300': 1.0}},
}
POLICY = {
    'max_weight': 0.85, 'max_group_weight': 0.85, 'max_equity_weight': 0.95,
    'max_gross_weight': 0.95, 'covariance_days': 60, 'annualization_days': 252,
    'vol_target': 0.99, 'volatility_policy': 'observe_only',
    'rebalance_band': 0.03, 'max_daily_turnover': 0.3,
}
SYMBOLS = ('SH515880', 'SH518880', 'SH510300')


def build_selector(prices, trend_window=2):
    trace = []
    factory = weekly_overlay_allocator(prices, {'trend_window': trend_window, 'vol_target': None}, trace)
    return factory(CANDIDATE, list(prices.index), prices.index[0]), trace


def replay(select, prices, start=None, eligible=SYMBOLS):
    """按日重放 select，模拟现金回放调用顺序，返回每日目标及是否与传入持仓一致。"""
    returns = prices.pct_change(fill_method=None)
    current, daily = {}, {}
    index = prices.index[prices.index >= (start or prices.index[0])]
    for date in index[:-1]:
        targets = select(pd.Series(dtype=float), list(eligible), dict(current), {},
                         returns.loc[:date], POLICY, exiting=set())
        daily[date] = {'targets': dict(targets), 'same_as_current': dict(targets) == dict(current)}
        current = dict(targets) if targets else current
    return daily


class WeeklyCadenceTest(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range('2021-12-06', '2022-01-07')
        self.dates = dates
        self.prices = pd.DataFrame(
            {symbol: [10 + index * .1 for index in range(len(dates))] for symbol in SYMBOLS}, index=dates)

    def test_decides_on_first_session_of_each_week_only(self):
        select, trace = build_selector(self.prices)
        start = pd.Timestamp('2021-12-13')
        daily = replay(select, self.prices, start=start)
        decision_days = [date for date in daily if not daily[date]['same_as_current']]
        self.assertEqual([str(date.date()) for date in decision_days], [row['date'] for row in trace])
        weeks = [date.to_period('W-SUN') for date in daily]
        self.assertEqual(len(trace), len(set(weeks)))
        for week in set(weeks):
            members = [date for date in daily if date.to_period('W-SUN') == week]
            decided = [date for date in decision_days if date in members]
            self.assertEqual(decided, [members[0]], f'{week} 决策日应为该周首个观察交易日')

    def test_hold_days_keep_previous_targets_and_first_buy_respects_budget(self):
        select, _ = build_selector(self.prices)
        start = pd.Timestamp('2021-12-13')
        daily = replay(select, self.prices, start=start)
        monday = start
        first = daily[monday]['targets']
        self.assertTrue(first, '首个决策日应产生目标')
        self.assertLessEqual(set(first), {'SH515880', 'SH518880'})
        self.assertAlmostEqual(sum(first.values()), 0.3, places=9, msg='首日建仓受30%换手预算约束')
        tuesday = self.dates[self.dates.get_loc(monday) + 1]
        self.assertTrue(daily[tuesday]['same_as_current'], '周中必须保持目标不变')

    def test_trend_gate_moves_failed_symbols_to_cash(self):
        dates = pd.bdate_range('2021-12-06', '2021-12-13')
        prices = pd.DataFrame({'SH515880': [10, 10, 10, 10, 10, 9.5],
                               'SH518880': [10, 10, 10, 10, 10, 10.5],
                               'SH510300': [10] * 6}, index=dates)
        select, trace = build_selector(prices)
        targets = select(pd.Series(dtype=float), list(SYMBOLS), {}, {},
                         prices.pct_change(fill_method=None), POLICY, exiting=set())
        self.assertEqual(trace[-1]['date'], str(dates[-1].date()))
        self.assertNotIn('SH515880', targets, '跌破两日均价的资产应转现金')
        self.assertIn('SH518880', targets)


class CadenceAuditTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2021-12-20', '2022-01-07')
        self.trace = [{'date': str(self.dates[position].date())} for position in (0, 5, 10)]

    def test_accepts_weekly_decisions_and_next_day_trades(self):
        trades = pd.DataFrame([{'date': str(self.dates[1].date())}, {'date': str(self.dates[6].date())}])
        result = cadence_audit(self.dates, self.dates[0], self.trace, trades)
        self.assertEqual(result['decision_days'], 3)
        self.assertTrue(result['trades_only_after_decisions'])

    def test_rejects_trades_outside_decision_follow_up(self):
        trades = pd.DataFrame([{'date': str(self.dates[2].date())}])
        with self.assertRaisesRegex(ValueError, '非决策日出现成交'):
            cadence_audit(self.dates, self.dates[0], self.trace, trades)

    def test_rejects_missing_week(self):
        trace = [{'date': str(self.dates[0].date())}]
        with self.assertRaisesRegex(ValueError, '决策日必须为每个自然周'):
            cadence_audit(self.dates, self.dates[0], trace, pd.DataFrame())


class GoalCheckTest(unittest.TestCase):
    GOALS = {'conservative': {'cagr': 0.05, 'max_drawdown': 0.10},
             'advanced': {'cagr': 0.10, 'max_drawdown': 0.30}}

    def test_conservative_and_advanced_flags(self):
        rows = [({'cagr': 0.062, 'max_drawdown': 0.095}, (True, False)),
                ({'cagr': 0.048, 'max_drawdown': 0.095}, (False, False)),
                ({'cagr': 0.062, 'max_drawdown': 0.105}, (False, False)),
                ({'cagr': 0.120, 'max_drawdown': 0.220}, (False, True)),
                ({'cagr': 0.120, 'max_drawdown': 0.310}, (False, False))]
        for metrics, (conservative, advanced) in rows:
            checks = goal_checks(metrics, self.GOALS)
            self.assertEqual(checks['conservative_met'], conservative, metrics)
            self.assertEqual(checks['advanced_met'], advanced, metrics)


if __name__ == '__main__':
    unittest.main()
