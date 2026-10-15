"""正式口径：冻结输入复现、时效门槛、资金约束、周频及 API 验证。"""
import asyncio
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
import pandas as pd
from backend.main import app
from backend import strategy_service
from src.formal_rules import ROOT, registry, weekly_sessions, trend_targets
from src.formal_service import stats, runtime_status, preview, verify_execution
from src.formal_adjusted import simulate
from src.formal_cash import simulate_cash, audit_cash_ledger
from src.market import load_market
from src.formal_orders import plan_orders
from src.portfolio import plan_orders as legacy_orders


class FormalRulesTests(unittest.TestCase):
    def test_weekly_holiday_first_session(self):
        dates = pd.to_datetime(['2026-09-28', '2026-09-29', '2026-09-30', '2026-10-09', '2026-10-12'])
        self.assertEqual(list(weekly_sessions(dates, dates[0]).strftime('%Y-%m-%d')),
                         ['2026-09-28', '2026-10-09', '2026-10-12'])

    def test_annual_dd_includes_previous_year_end(self):
        index = pd.to_datetime(['2016-09-12', '2016-12-30', '2017-01-03', '2017-12-29', '2018-01-02'])
        metrics, _ = stats(pd.DataFrame({'account': [100, 200, 150, 180, 180]}, index=index), 100)
        year = next(a for a in metrics['annual'] if a['year'] == 2017)
        self.assertAlmostEqual(year['return'], -.1)
        self.assertAlmostEqual(year['max_drawdown'], .25)
        self.assertTrue(year['complete'])
        self.assertFalse(metrics['annual'][0]['complete'])

    def test_stale_and_elapsed_signals_never_execute(self):
        p = {'data_end': '2026-10-12', 'advice': {'signal_date': '2026-10-12'}}
        r = runtime_status(p, pd.Timestamp('2026-10-13 08:00', tz='Asia/Shanghai'))
        self.assertEqual(r['advice_status'], 'pending')
        r = runtime_status(p, pd.Timestamp('2026-10-13 11:00', tz='Asia/Shanghai'))
        self.assertEqual(r['advice_status'], 'elapsed')
        self.assertFalse(r['can_execute'])
        r = runtime_status(p, pd.Timestamp('2026-10-13 20:00', tz='Asia/Shanghai'))
        self.assertTrue(r['stale'])
        self.assertEqual(r['next_signal_date'], '2026-10-19')

    def test_trend_failure_does_not_fallback_to_csi300(self):
        definition = registry()['strategies'][1]
        dates = pd.bdate_range('2024-01-01', periods=210)
        prices = pd.DataFrame({'SH515880': np.arange(300, 90, -1), 'SH510300': np.arange(1, 211),
                               'SH518880': np.arange(1, 211)}, index=dates)
        ret = prices.pct_change(fill_method=None)
        allowed = list(prices.columns)
        targets = trend_targets(definition['candidate'], dates[-1], allowed, {}, ret, prices, definition['policy'])
        self.assertNotIn('SH515880', targets)
        self.assertNotIn('SH510300', targets)
        targets = trend_targets(definition['candidate'], dates[-1], allowed[1:], {}, ret, prices, definition['policy'])
        self.assertIn('SH510300', targets)

    def test_production_modules_have_no_research_imports(self):
        for p in (ROOT / 'src').glob('formal_*.py'):
            self.assertNotRegex(p.read_text(encoding='utf-8'), r'(?m)^\s*(?:from|import) research[. ]')

    def test_unchanged_target_never_generates_rounding_order(self):
        prices, quantities, cash = {'SH510300': 4.36}, {'SH510300': 34694.12}, 100000
        weight = quantities['SH510300'] * 4.36 / (cash + quantities['SH510300'] * 4.36)
        self.assertEqual(plan_orders({'SH510300': weight}, quantities, cash, prices,
                                   {'SH510300': 1e8}, registry()['strategies'][0]['policy']), [])

    def test_reduction_and_exit_produce_integer_shares(self):
        policy = registry()['strategies'][0]['policy']
        for target in (.1, 0):
            orders = plan_orders({'SH510300': target}, {'SH510300': 32094.12}, 100000,
                                 {'SH510300': 4.36}, {'SH510300': 1e8}, policy)
            self.assertTrue(orders)
            self.assertTrue(all(o['quantity'] % 1 == 0 for o in orders))

    def test_proxy_split_sub_lot_exit_clears_the_entire_position(self):
        dates = pd.bdate_range('2016-09-09', '2016-09-21')
        factor = pd.Series(np.where(dates < pd.Timestamp('2016-09-15'), 1., .5498), index=dates)
        frame = lambda values: pd.DataFrame({'SH510300': values}, index=dates)
        fields = {'adj_factor': frame(factor), 'adj_close': frame(1.),
                  'close': frame(1 / factor), 'open': frame(1 / factor), 'volume': frame(1e8)}
        market = SimpleNamespace(calendar=dates, wide=lambda key: fields[key],
            eligibility=pd.DataFrame({'date': dates, 'symbol': 'SH510300', 'eligible': True, 'avg_amount': 1e8}))
        policy = {**registry()['strategies'][0]['policy'], 'portfolio_style': 'rotation',
                  'topk': 1, 'initial_capital': 1000.}
        scores = pd.Series([1., 1.], index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp('2016-09-12'), 'SH510300'), (pd.Timestamp('2016-09-19'), 'SH510300')],
            names=['datetime', 'instrument']))
        state = {}
        report, trades, _ = simulate(market, scores, policy, unavailable_dates={pd.Timestamp('2016-09-19')}, state=state)
        self.assertEqual(state['quantities'], {})
        self.assertTrue(any((trades.side == 'sell') & (trades.quantity < 100)))
        self.assertEqual(report.value.iloc[-1], 0)


class FrozenParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'output/history_backfill_2016/side_20261013'
        if not source.exists():
            raise unittest.SkipTest('本地历史基线不可用')
        cls.market = load_market(source / 'market_2013_2026.duckdb')
        cls.spec = registry()
        cls.source = source

    def test_adjusted_account_exact_historical_parity(self):
        pred = pd.read_pickle(self.source / 'predictions_lightgbm.pkl')
        days = weekly_sessions(self.market.calendar, self.spec['start'])
        scores = pred[pred.index.get_level_values('datetime').isin(days)]
        observed = []
        report, trades, _ = simulate(self.market, scores, self.spec['strategies'][0]['policy'],
            observer=lambda *args: observed.append(args[0]), order_planner=legacy_orders)
        old = pd.read_csv(ROOT / 'output/two_line_research/visual_comparison_backfilled_20261013/report_lightgbm_weekly_backfilled.csv',
                          index_col='date', parse_dates=True)
        np.testing.assert_allclose(report.account, old.account, rtol=1e-12, atol=.00001)
        verify_execution(self.market, report, trades, self.spec['start'])
        self.assertEqual(list(pd.DatetimeIndex(observed).strftime('%Y-%m-%d')), list(days.strftime('%Y-%m-%d')))

    def test_cash_engine_parity_on_same_inputs(self):
        # 测试可以导入冻结研究；正式 src 不允许依赖研究实现。
        from research.goal30.cash_replay import simulate_cash as frozen
        from research.goal30.weekly_goal import weekly_overlay_allocator
        definition = self.spec['strategies'][1]
        actions = json.loads((ROOT / 'strategy_corporate_actions.json').read_text(encoding='utf-8'))['actions']
        factory = weekly_overlay_allocator(self.market.wide('adj_close'), {'trend_window': 200, 'vol_target': .15}, [])
        with patch('research.goal30.cash_replay.allocator', factory), patch('research.goal30.cash_replay.plan_orders', factory.plan):
            expected, _, _, _, _ = frozen(self.market, definition['candidate'], definition['policy'], self.spec['start'], actions)
        def select(scores, allowed, current, days, returns, policy, *, exiting=()):
            return trend_targets(definition['candidate'], returns.index[-1], allowed, current, returns,
                                 self.market.wide('adj_close'), policy, exiting)
        report, trades, events, _, _ = simulate_cash(self.market, definition['candidate'], definition['policy'],
                                                    self.spec['start'], actions, selector=select, order_planner=legacy_orders)
        np.testing.assert_allclose(report.account, expected.account, rtol=1e-12, atol=.00001)
        self.assertLessEqual(audit_cash_ledger(self.market, report, trades, events, 1e6)['max_nav_error'], .01)
        verify_execution(self.market, report, trades, self.spec['start'])

    def test_unavailable_model_exits_account_without_fractional_ghost_holdings(self):
        market = copy.copy(self.market)
        market.calendar = market.calendar[market.calendar <= pd.Timestamp('2016-10-14')]
        market.bars = market.bars[market.bars.date.le(market.calendar[-1])]
        pred = pd.read_pickle(self.source / 'predictions_lightgbm.pkl')
        days = weekly_sessions(market.calendar, self.spec['start'])
        scores = pred[pred.index.get_level_values('datetime').isin(days)]
        state = {}
        report, trades, _ = simulate(market, scores, self.spec['strategies'][0]['policy'],
                                    unavailable_dates={days[-1]}, state=state)
        self.assertEqual(state['quantities'], {})
        self.assertAlmostEqual(report.value.iloc[-1], 0)
        verify_execution(market, report, trades, self.spec['start'])


class FormalAPITests(unittest.TestCase):
    def request(self, method, url, **kwargs):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8000') as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(run())

    def test_refresh_csrf_guard_and_duplicate_task(self):
        self.assertEqual(self.request('POST', '/api/strategies/refresh', json={}).status_code, 403)
        self.assertEqual(self.request('POST', '/api/strategies/refresh', json={}, headers={
            'X-Local-Request': '1', 'Origin': 'https://example.com'}).status_code, 403)
        with patch.object(strategy_service, 'start', side_effect=RuntimeError('已有刷新')):
            self.assertEqual(self.request('POST', '/api/strategies/refresh', json={}, headers={'X-Local-Request': '1'}).status_code, 409)

    def test_invalid_real_account_is_rejected(self):
        base = {'run_id': '20261013-110620-97bb6708', 'cash': -1, 'holdings': []}
        self.assertEqual(self.request('POST', '/api/strategies/lightgbm/preview', json=base,
                                     headers={'X-Local-Request': '1'}).status_code, 422)
        base['cash'] = 1000
        base['holdings'] = [{'symbol': 'SH510300', 'quantity': 100}] * 2
        self.assertEqual(self.request('POST', '/api/strategies/lightgbm/preview', json=base,
                                     headers={'X-Local-Request': '1'}).status_code, 422)

    def test_published_snapshot_and_preview(self):
        if not (ROOT / 'output/strategies/current.json').exists():
            self.skipTest('尚无正式快照')
        result = self.request('GET', '/api/strategies')
        self.assertEqual(result.status_code, 200)
        snap = result.json()
        self.assertEqual(len(snap['strategies']), 2)
        for s in snap['strategies']:
            self.assertEqual(s['series']['dates'][0], '2016-09-12')
            self.assertEqual(s['series']['capital'][0], 1e6)
            self.assertEqual(s['run_id'], snap['run_id'])
            r = self.request('POST', f'/api/strategies/{s["id"]}/preview', json={
                'run_id': snap['run_id'], 'cash': 1e6, 'holdings': []}, headers={'X-Local-Request': '1'})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertLessEqual(sum(x['estimated_amount'] for x in r.json()['rows'] if x['side'] == 'buy'), 1e6)
            self.assertTrue(all(x['quantity'] % 100 == 0 for x in r.json()['rows'] if x['side'] == 'buy'))
            bad = self.request('POST', f'/api/strategies/{s["id"]}/preview', json={
                'run_id': snap['run_id'], 'cash': 1e6, 'holdings': [{'symbol': 'SH513100', 'quantity': 100}]},
                headers={'X-Local-Request': '1'})
            self.assertEqual(bad.status_code, 400)
