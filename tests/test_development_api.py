"""开发后台使用新正式快照：版本一致、真实排名、趋势替代和损坏拒绝。"""
import asyncio
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI

from backend import development_service as service
from backend.development_api import router
from src.formal_rules import registry, SYMBOLS
from src.runs import sha256


class DevelopmentFixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.out = self.root / 'output/strategies'
        self.run_id = '20261013-114948-477168d8'
        self.other_id = '20261013-120000-12345678'
        for name, path in (('ROOT', self.root), ('OUT', self.out), ('DATABASE', self.root / 'formal.duckdb')):
            self.stack.enter_context(patch.object(service, name, path))
        service._cache.clear()
        service._markets.clear()
        self.addCleanup(service._cache.clear)
        self.addCleanup(service._markets.clear)
        self.calendar = pd.bdate_range(end='2026-10-12', periods=220)
        symbols = ['SH510300', 'SZ159915', 'SH512880', 'SH511260', 'SH518880', 'SH515880', 'SH513100']
        self.adjusted = pd.DataFrame({s: np.arange(1., 221.) for s in symbols}, index=self.calendar)
        self.adjusted['SH515880'] = np.arange(300., 80., -1.)
        self.close = self.adjusted / 2
        self.eligibility = pd.DataFrame([
            {'date': day, 'symbol': s, 'eligible': s not in ['SH512880', 'SH513100']}
            for day in self.calendar for s in symbols])
        self.market = SimpleNamespace(calendar=self.calendar, fingerprint='registered-fingerprint',
            eligibility=self.eligibility, wide=lambda field: self.close if field == 'close' else self.adjusted)
        self.market_loader = self.stack.enter_context(patch.object(service, 'load_market', return_value=self.market))
        self.definitions = registry()
        self.dates = ['2026-09-28', '2026-09-29', '2026-09-30', '2026-10-09', '2026-10-12']
        self.write_run(self.run_id)
        self.publish(self.run_id)
        self.app = FastAPI()
        self.app.include_router(router)

    def write_json(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')

    def write_run(self, run_id):
        directory = self.out / 'runs' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        self.write_json(directory / 'registration.json', {
            'registry': self.definitions, 'market_fingerprint': self.market.fingerprint,
            'data_audit': {'end': self.dates[-1], 'bars': 123, 'nav_availability': 'ann_date < signal_date'}})
        for id_ in service.STRATEGY_IDS:
            definition = next(d for d in self.definitions['strategies'] if d['id'] == id_)
            payload = {'id': id_, 'run_id': run_id, 'name': id_, 'data_end': self.dates[-1],
                'series': {'dates': self.dates, 'capital': [1e6] * len(self.dates)},
                'metrics': {'cagr': .05}, 'validation': {'weekly_t_plus_1': True},
                'advice': {'signal_date': self.dates[-1]}, 'weekly_holdings': []}
            if definition.get('kind') == 'valuation_bands':
                payload.update(kind='valuation_bands', valuation_data_end=self.dates[-1])
                pd.DataFrame([{'session': day, 'value': 20., 'valuation_date': day,
                    'valuation_available_at': day + ' 19:00', 'valuation_lag': 1,
                    'L1': 21., 'L2': 19., 'L3': 17., 'S': 25., 'phase_before': 'holding',
                    'buy_steps_before': 1, 'reason': 'hold_cooldown', 'detail': '冷却期保持',
                    'eligible': True, 'etf_close': 4.} for day in self.dates]).to_csv(
                    directory / (id_ + '_signals.csv'), index=False)
            if definition.get('kind') == 'index_benchmark':
                payload.update(kind='index_benchmark', benchmark={'index_code':'H20269','close':104.})
                index_dir=directory/'index_data'
                index_dir.mkdir(exist_ok=True)
                pd.DataFrame({'date':self.dates,'close':range(100,105),'index_code':'H20269'}).to_csv(index_dir/'H20269.csv',index=False)
                self.write_json(index_dir/'manifest.json',{'index_code':'H20269'})
            if definition.get('kind') == 'dividend_band':
                payload.update(kind='dividend_band',capabilities={'account_preview':False,'executable_orders':False})
                pd.DataFrame([{'session':day,'dividend_yield_pct':4.9,'implied_yield':.049,
                    'red_target':.85,'bond_target':.10,'cash_target':.05} for day in self.dates]).to_csv(
                    directory/(id_+'_signals.csv'),index=False)
                pd.DataFrame([{'date':day,'nav':1e6,'red_value':6e5,'bond_value':3e5,'cash':1e5}
                    for day in self.dates]).to_csv(directory/(id_+'_daily.csv'),index=False)
                dividend_dir=directory/'dividend_data'
                dividend_dir.mkdir(exist_ok=True)
                pd.DataFrame({'date':self.dates,'red':range(100,105),'price':range(90,95)}).to_csv(dividend_dir/'panel.csv',index=False)
                self.write_json(dividend_dir/'manifest.json',{'kind':'dividend_band'})
            self.write_json(directory / (id_ + '.json'), payload)
            self.write_json(directory / (id_ + '_context.json'), {
                'date': self.dates[-1], 'eligible': ['SH510300', 'SH518880', 'SH515880', 'SH511260'],
                'prices': self.close.iloc[-1].to_dict()})
            pd.DataFrame([{'date': '2026-09-29', 'signal_date': '2026-09-28', 'symbol': 'SH510300',
                           'side': 'buy', 'quantity': 100, 'price': 3., 'optional': np.nan}]).to_csv(
                directory / (id_ + '_trades.csv'), index=False)
        self.rehash(run_id)

    def rehash(self, run_id):
        directory = self.out / 'runs' / run_id
        self.write_json(directory / 'completion.json', {'run_id': run_id, 'completed_at': '2026-10-13T11:50:00+08:00',
            'artifact_hashes': {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob('*') if p.is_file() and p.name != 'completion.json'},
            'refresh_error': None})

    def publish(self, run_id):
        self.write_json(self.out / 'current.json', {'run_id': run_id,
            'directory': f'output/strategies/runs/{run_id}', 'data_end': self.dates[-1]})

    def request(self, url):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://127.0.0.1') as client:
                return await client.get(url)
        return asyncio.run(run())


class DevelopmentServiceTests(DevelopmentFixture):
    def test_three_strategies_load_without_model_files(self):
        with patch.object(pd, 'read_pickle', side_effect=AssertionError('禁止加载已停用模型')):
            overview = service.overview()
            self.assertEqual([s['id'] for s in overview['strategies']], list(service.STRATEGY_IDS))
            result = service.get_picks('csi300_pe_equal')
            self.assertEqual(result['kind'], 'valuation')
            self.assertEqual(len(result['rows']), 1)
            self.assertIsNone(result['rows'][0]['score'])
            self.assertIsNone(result['rows'][0]['trend_pass'])
            self.assertEqual(result['valuation']['entry_upper'], 21.)
            self.assertEqual(len(service.get_universe('csi300_pe_equal')['rows']), 1)
            self.assertEqual(service.get_picks('dividend_band_a')['kind'], 'dividend_band')

    def test_overview_uses_new_registry_and_both_same_run(self):
        result = service.overview()
        self.assertEqual(result['source'], 'formal_weekly')
        self.assertEqual(result['registry'], self.definitions)
        self.assertEqual(result['data_audit']['bars'], 123)
        self.assertTrue(all(s['run_id'] == result['run_id'] for s in result['strategies']))
        # 调用方修改结果不能污染正式缓存。
        result['strategies'][0]['metrics']['cagr'] = 100
        self.assertEqual(service.overview()['strategies'][0]['metrics']['cagr'], .05)

    def test_run_binding_survives_pointer_switch(self):
        old = service.overview()['run_id']
        self.write_run(self.other_id)
        self.publish(self.other_id)
        self.assertEqual(service.overview()['run_id'], self.other_id)
        self.assertEqual(service.backtest('trend200_vol15', old)['run_id'], old)
        self.assertEqual(service.available_dates('trend200_vol15', old)['run_id'], old)
        self.assertEqual(service.get_picks('trend200_vol15', run_id=old)['run_id'], old)

    def test_weekly_dates_include_holiday_first_session_and_fallback(self):
        self.assertEqual(service.available_dates('trend200_vol15')['dates'],
                         ['2026-09-28', '2026-10-09', '2026-10-12'])
        result = service.get_picks('trend200_vol15', date='2026-09-30')
        self.assertEqual(result['date'], '2026-09-28')
        self.assertEqual(result['requested_date'], '2026-09-30')
        self.assertTrue(result['fallback'])
        result = service.get_picks('trend200_vol15', date='2026-10-10')
        self.assertEqual(result['date'], '2026-10-09')
        self.assertTrue(result['fallback'])

    def test_retired_strategies_rejected_before_reading_bundle(self):
        with patch.object(service, '_bundle', side_effect=AssertionError('停用ID不可加载快照')):
            for sid in ['lightgbm','chinext_pb_equal','dividend_lowvol_total_return']:
                for fn in [service.get_picks, service.get_universe, service.available_dates, service.backtest]:
                    with self.assertRaises(KeyError):fn(sid)

    def test_trend_failure_does_not_activate_proxy_and_sma_is_raw_unit(self):
        result = service.get_picks('trend200_vol15')
        rows = {r['symbol']: r for r in result['rows']}
        self.assertFalse(rows['SH515880']['trend_pass'])
        self.assertTrue(rows['SH510300']['trend_pass'])
        self.assertFalse(rows['SH510300']['fallback_active'])
        self.assertEqual(rows['SH510300']['candidate_weight'], 0)
        self.assertAlmostEqual(rows['SH510300']['sma200'], self.adjusted.SH510300.tail(200).mean() / 2)
        self.assertAlmostEqual(rows['SH510300']['deviation'],
                               self.adjusted.SH510300.iloc[-1] / self.adjusted.SH510300.tail(200).mean() - 1)
        self.assertTrue(all(r['score'] is None and r['rank'] is None for r in rows.values()))

    def test_dividend_band_uses_selected_historical_signal_without_future_market(self):
        result = service.get_picks('dividend_band_a', date='2026-09-30')
        self.assertEqual(result['date'], '2026-09-28')
        self.assertEqual(result['dividend_signal']['implied_yield'], .049)
        self.assertEqual(result['dividend_signal']['date'], result['date'])
        self.assertEqual(result['dividend_signal']['red_target'], .85)
        self.assertTrue(all(r['quantity'] is None for r in result['rows']))
        self.market_loader.assert_not_called()

    def test_only_qualification_failure_can_activate_proxy(self):
        self.eligibility.loc[self.eligibility.symbol.eq('SH515880'), 'eligible'] = False
        rows = {r['symbol']: r for r in service.get_picks('trend200_vol15')['rows']}
        self.assertTrue(rows['SH510300']['fallback_active'])
        self.assertEqual(rows['SH510300']['candidate_weight'], .6)
        self.eligibility.loc[self.eligibility.symbol.eq('SH510300'), 'eligible'] = False
        rows = {r['symbol']: r for r in service.get_picks('trend200_vol15')['rows']}
        self.assertEqual(rows['SH510300']['candidate_weight'], 0)

    def test_incomplete_trend_window_stays_cash(self):
        date = self.calendar[10]
        rows = service._trend_rows(service._bundle(), self.market, date)
        self.assertTrue(all(not r['trend_pass'] and r['sma200'] is None and r['candidate_weight'] == 0 for r in rows))

    def test_backtest_trade_rows_belong_to_requested_strategy_without_model_folds(self):
        pe = service.backtest('csi300_pe_equal')
        trend = service.backtest('trend200_vol15')
        self.assertEqual(pe['trades'][0]['quantity'], 100)
        self.assertIsNone(pe['trades'][0]['optional'])
        self.assertEqual(pe['folds'], [])
        self.assertEqual(trend['folds'], [])
        self.assertEqual(trend['definition']['id'], 'trend200_vol15')
        self.assertEqual(pe['validation'], pe['strategy']['validation'])

    def test_universe_has_one_pe_three_trend_and_no_tradable_index(self):
        pe = service.get_universe('csi300_pe_equal')
        trend = service.get_universe('trend200_vol15')
        self.assertEqual({r['symbol'] for r in pe['rows']}, {'SH510300'})
        self.assertEqual({r['symbol'] for r in trend['rows']}, {'SH515880', 'SH518880', 'SH510300'})
        self.assertTrue(next(r for r in trend['rows'] if r['symbol'] == 'SH515880')['eligible'])
        dividend = service.get_universe('dividend_band_a')['rows']
        self.assertEqual({r['symbol'] for r in dividend}, {'H20269','SH511260'})
        self.assertTrue(all(not r['tradeable'] and r['equivalent'] for r in dividend))

    def test_tampered_cached_artifact_is_rejected(self):
        service.overview()
        path = self.out / 'runs' / self.run_id / 'trend200_vol15.json'
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(service.DevelopmentUnavailableError, '哈希校验失败'):
            service.overview()

    def test_market_history_revision_cannot_fill_published_predictions(self):
        self.market.fingerprint = 'changed-history'
        with self.assertRaisesRegex(service.DevelopmentUnavailableError, '登记历史已改变'):
            service.get_picks('trend200_vol15')

    def test_bad_pointer_and_mixed_payload_are_rejected(self):
        self.write_json(self.out / 'current.json', {'run_id': self.run_id, 'directory': '../unrelated', 'data_end': self.dates[-1]})
        with self.assertRaises(service.DevelopmentUnavailableError):
            service.overview()
        self.publish(self.run_id)
        path = self.out / 'runs' / self.run_id / 'trend200_vol15.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['run_id'] = self.other_id
        self.write_json(path, payload)
        self.rehash(self.run_id)
        with self.assertRaisesRegex(service.DevelopmentUnavailableError, '混用了运行'):
            service.overview()


class DevelopmentAPITests(DevelopmentFixture):
    def test_retired_strategy_api_queries_are_rejected(self):
        for sid in ['lightgbm','chinext_pb_equal','dividend_lowvol_total_return']:
            for endpoint in ['picks','universe','picks/dates','backtest']:
                self.assertEqual(self.request('/api/development/'+endpoint+'?strategy_id='+sid).status_code,422)

    def test_all_endpoints_use_formal_weekly_and_one_run(self):
        paths = ['overview']
        for definition in self.definitions['strategies']:
            paths.extend(endpoint + '?strategy_id=' + definition['id']
                         for endpoint in ['universe', 'picks/dates', 'picks', 'backtest'])
        for path in paths:
            result = self.request('/api/development/' + path)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()['run_id'], self.run_id)

    def test_invalid_query_is_bounded(self):
        for query in ['strategy_id=bad', 'strategy_id=trend200_vol15&topk=0', 'strategy_id=trend200_vol15&topk=30',
                      'strategy_id=trend200_vol15&run_id=../../old', 'strategy_id=trend200_vol15&date=not-a-date']:
            self.assertEqual(self.request('/api/development/picks?' + query).status_code, 422)
        self.assertEqual(self.request('/api/development/picks?strategy_id=trend200_vol15&date=2026-02-30').status_code, 400)
        self.assertEqual(self.request('/api/development/picks?strategy_id=trend200_vol15&date=2016-01-01').status_code, 400)

    def test_missing_snapshot_is503_and_never_falls_back_to_legacy(self):
        (self.out / 'current.json').unlink()
        # 即使旧指针存在，也不读取。
        self.write_json(self.root / 'output/current_run.json', {'run_id': 'old'})
        result = self.request('/api/development/overview')
        self.assertEqual(result.status_code, 503)
        self.assertIn('正式周频', result.json()['detail'])

    def test_bound_read_can_still_use_verified_old_run_after_refresh(self):
        self.write_run(self.other_id)
        self.publish(self.other_id)
        result = self.request('/api/development/backtest?strategy_id=trend200_vol15&run_id=' + self.run_id)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['run_id'], self.run_id)

    def test_unavailable_formal_database_is503(self):
        self.market_loader.side_effect = duckdb.IOException('正式数据库暂不可读')
        result = self.request('/api/development/picks?strategy_id=trend200_vol15')
        self.assertEqual(result.status_code, 503)
        self.assertIn('历史输入核对失败', result.json()['detail'])


if __name__ == '__main__':
    unittest.main()
