"""等权估值正式迁移：冻结净值对照、现金账本、数据修订和时效约束。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from src import formal_valuation as valuation
from src.formal_rules import registry
from src.formal_service import stats, next_session, runtime_status, preview, snapshot_needs_refresh
from src.valuation.data import load_snapshot


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.to_datetime(['2011-01-04', '2011-01-05', '2011-01-06', '2011-01-07', '2011-01-10'])
        self.frames = {
            'index_price': pd.DataFrame({'index_id': ['000300'] * 2, 'trade_date': self.calendar[:2], 'close': [100., 99.]}),
            'index_valuation': pd.DataFrame({'index_id': ['000300'] * 2, 'metric': ['PE_TTM_EQUAL'] * 2,
                                            'trade_date': self.calendar[:2], 'value': [20., 19.]})}
        self.incoming = pd.DataFrame({'date': self.calendar[:4], 'close': [100., 99., 101., 102.],
                                       'equalWeightedPeTtm': [20., 19., 21., 22.]})

    def test_append_keeps_frozen_overlap_and_next_session_availability(self):
        valuation.extend_index_frames(self.frames, '000300', self.incoming, self.calendar, 'hash', '2026-11-09')
        values = self.frames['index_valuation']
        self.assertEqual(values.value.tolist(), [20., 19., 21., 22.])
        self.assertEqual(values.iloc[2].effective_available_at, pd.Timestamp('2011-01-07 19:00'))
        self.assertEqual(values.iloc[3].effective_available_at, pd.Timestamp('2011-01-10 19:00'))

    def test_revised_history_and_missing_session_are_rejected_without_mutation(self):
        for incoming in [self.incoming.assign(equalWeightedPeTtm=[20.1, 19., 21., 22.]), self.incoming.drop(index=2)]:
            frames = copy.deepcopy(self.frames)
            with self.assertRaises(ValueError):
                valuation.extend_index_frames(frames, '000300', incoming, self.calendar, 'hash', '2026-11-09')
            pd.testing.assert_frame_equal(frames['index_valuation'], self.frames['index_valuation'])

    def test_provider_precomputed_percentiles_are_not_used(self):
        packet = '1:' + json.dumps({'indexDailyItems': [{'date': '2011-01-04', 'close': 100,
                         'equalWeightedPeTtm': 20, 'historyPePercentile': .99}]})
        frame = valuation.extract_daily('<script>self.__next_f.push(' + json.dumps([1, packet]) + ')</script>')
        self.assertEqual(frame.iloc[0].equalWeightedPeTtm, 20)
        with self.assertRaises(ValueError):
            valuation.extract_daily('<html>需要登录</html>')


class FormalValuationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not valuation.SEED.exists():
            raise unittest.SkipTest('缺少本地真实估值种子')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.directory = Path(cls.tmp.name)
        cls.data = load_snapshot(valuation.SEED)
        cls.spec = registry()
        cls.payloads = {}
        for definition in cls.spec['strategies']:
            if definition.get('kind') == 'valuation_bands':
                cls.payloads[definition['id']] = valuation.build_strategy(
                    cls.directory, definition, cls.spec, cls.data, '20261109-120000-12345678',
                    pd.Timestamp('2026-11-04'), stats, next_session)

    def test_promoted_accounts_match_independently_verified_frozen_research(self):
        golden = valuation.ROOT / 'output/valuation_bands/runs/real-etfrun-equal-v2/cases'
        for sid, case in [('csi300_pe_equal', 'A_pe_weekly')]:
            if not (golden / case / 'daily_account.csv').exists():
                self.skipTest('本地冻结研究对照不可用')
            expected = pd.read_csv(golden / case / 'daily_account.csv')
            actual = pd.read_csv(self.directory / 'valuation_cases' / sid / 'daily_account.csv')
            self.assertEqual(expected.session.tolist(), actual.session.tolist())
            for column in ('nav', 'cash', 'dividend_receivable', 'equity_value'):
                np.testing.assert_allclose(actual[column], expected[column], atol=1e-6, rtol=0)

    def test_payloads_have_verified_ledger_history_and_equal_weight_limits(self):
        for sid, payload in self.payloads.items():
            self.assertEqual(payload['validation']['ledger']['status'], 'passed')
            self.assertEqual(payload['metrics']['worst_complete_year']['year'], 2018)
            self.assertFalse(payload['investment_admission'])
            self.assertFalse(payload['goals']['worst_complete_year'])
            self.assertEqual(len(payload['series']['dates']), 2460)
            self.assertEqual(len(payload['weekly_holdings']), 10)
            self.assertEqual(len(payload['weekly_allocation']), 50)
            self.assertEqual(payload['weekly_allocation'][:10], payload['weekly_holdings'])
            for week in payload['weekly_allocation']:
                self.assertLessEqual(week['date'], payload['data_end'])
                self.assertAlmostEqual(sum(h['weight'] for h in week['holdings'])
                    + week['cash_weight'] + week['receivable_weight'], 1.)
            self.assertFalse(payload['capabilities']['account_preview'])
            self.assertIn('等权', payload['valuation_basis'])
            self.assertEqual(payload['valuation']['metric'], 'PE_TTM_EQUAL' if sid == valuation.IDS[0] else 'PB_ETFRUN_EQUAL')

    def test_valuation_staleness_blocks_even_if_etf_prices_are_current(self):
        payload = copy.deepcopy(self.payloads['csi300_pe_equal'])
        payload.update(data_end='2026-11-09', valuation_data_end='2026-11-04')
        state = runtime_status(payload, pd.Timestamp('2026-11-09 20:00', tz='Asia/Shanghai'))
        self.assertTrue(state['valuation_stale'])
        self.assertTrue(state['stale'])
        self.assertEqual(state['advice_status'], 'stale')

    def test_preview_rejects_missing_batch_history(self):
        definition = next(d for d in self.spec['strategies'] if d['id'] == valuation.IDS[0])
        with self.assertRaisesRegex(ValueError, '批次'):
            preview({}, definition, 1e6, [])

    def test_new_signal_order_quantity_uses_intent_instead_of_current_holdings(self):
        definition = next(d for d in self.spec['strategies'] if d['id'] == valuation.IDS[0])
        directory = self.directory / 'entry_signal'
        directory.mkdir()
        payload = valuation.build_strategy(directory, definition, self.spec, self.data,
            '20261109-120001-12345678', pd.Timestamp('2018-02-12'), stats, next_session)
        self.assertEqual(payload['advice']['orders_count'], 1)
        row = payload['advice']['rows'][0]
        self.assertEqual(row['side'], 'buy')
        self.assertGreater(row['quantity'], 0)
        self.assertEqual(row['quantity'] % 100, 0)
        self.assertLess(row['estimated_amount'], payload['advice']['cash'])
        self.assertGreater(row['target_weight'], 0)

    def test_failed_fetch_retains_valuation_and_reports_error_and_volume_units(self):
        last = pd.Timestamp('2026-11-05')
        bars = pd.DataFrame([{'date': last, 'symbol': s, 'open': 4., 'high': 4.1, 'low': 3.9,
                               'close': 4., 'volume': 1234., 'amount': 493600., 'source': 'tencent_raw'}
                             for s in ('SH510300', 'SZ159915')])
        market = SimpleNamespace(calendar=pd.DatetimeIndex([last]), bars=bars, fingerprint='test')
        directory = self.directory / 'failed_fetch'
        directory.mkdir()
        with patch.object(valuation.requests, 'get', side_effect=requests.ConnectionError('offline')):
            data, errors = valuation.prepare_data(directory, market, {'actions': []}, fetch=True)
        self.assertEqual(len(errors), 1)
        self.assertEqual(set(data.frames['index_valuation'].index_id), {'000300'})
        self.assertEqual(set(data.frames['etf_bars'].symbol), {'SH510300'})
        self.assertEqual(data.etf_bars('SH510300').iloc[-1].volume_shares, 123400.)
        pd.testing.assert_series_equal(data.index_valuation('000300', 'PE_TTM_EQUAL').value,
                                      self.data.index_valuation('000300', 'PE_TTM_EQUAL').value)


class StartupUpgradeTests(unittest.TestCase):
    def test_current_date_old_two_strategy_registration_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'output/strategies'
            run = out / 'runs/example'
            run.mkdir(parents=True)
            spec = registry()
            old = copy.deepcopy(spec)
            old['strategies'] = old['strategies'][:2]
            (run / 'registration.json').write_text(json.dumps({'registry': old}), encoding='utf-8')
            (out / 'current.json').write_text(json.dumps({'directory': 'output/strategies/runs/example',
                                                         'data_end': '2026-11-09'}), encoding='utf-8')
            with patch('src.formal_service.ROOT', root), patch('src.formal_service.OUT', out):
                self.assertTrue(snapshot_needs_refresh(pd.Timestamp('2026-11-09 20:00', tz='Asia/Shanghai')))
                (run / 'registration.json').write_text(json.dumps({'registry': spec}), encoding='utf-8')
                self.assertFalse(snapshot_needs_refresh(pd.Timestamp('2026-11-09 20:00', tz='Asia/Shanghai')))


if __name__ == '__main__':
    unittest.main()
