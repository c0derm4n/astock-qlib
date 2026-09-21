import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import exchange_calendars as xcals

from backend.artifacts import write_json_atomic
from research.goal30 import daily, forward
from src.account import apply_fills, initialize


class ForwardLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.account_path = self.root / 'account.json'
        self.opening = initialize(self.account_path, 10000.)
        self.observation = {'executable': False, 'orders': [], 'paper_orders': [
            {'symbol': 'SH515880', 'side': 'buy', 'quantity': 100., 'reference_price': 10.,
             'target_weight': .1, 'notional_limit': 1e6}],
            'observation_id': 'test-only', 'signal_date': '2026-09-10', 'execution_date': '2026-09-11',
            'account_nav': 10000., 'target_weights': {'SH515880': .1},
            'strategy_fingerprint': 'test-strategy', 'runtime_sha256': 'test-runtime',
            'checked_at': '2026-09-10T19:00:00+08:00'}
        self.decision = forward.paper_decision(self.observation, self.opening, self.account_path)
        self.signal_time = pd.Timestamp('2026-09-10 19:00', tz='Asia/Shanghai')
        self.close_time = pd.Timestamp('2026-09-11 19:00', tz='Asia/Shanghai')
        self.market = SimpleNamespace(calendar=pd.DatetimeIndex(['2026-09-11']), fingerprint='test-data',
            bars=pd.DataFrame({'date': [pd.Timestamp('2026-09-11')], 'symbol': ['SH515880'], 'close': [11.]}))

    def freeze(self):
        with patch.object(forward, '_now', return_value=self.signal_time):
            return forward.freeze(self.root, self.decision, self.opening)

    def filled_account(self, quantity=100., minute='30', decision=None):
        decision = decision or self.decision
        fill = {'execution_id': f'fill-{quantity}', 'decision_id': decision['decision_id'],
                'order_id': decision['orders'][0]['order_id'], 'symbol': 'SH515880', 'side': 'buy',
                'quantity': quantity, 'price': 10., 'fee': .3,
                'executed_at': f'2026-09-11T09:{minute}:00+08:00'}
        return apply_fills(self.account_path, decision, [fill])

    def test_standard_account_fill_import_and_close_qualify_one_day(self):
        self.freeze()
        account = self.filled_account()
        with patch.object(forward, '_now', return_value=self.close_time):
            close = forward.record_close(self.root, account, self.market)
            evidence = forward.evidence(self.root)
        self.assertTrue(close['qualified'])
        self.assertEqual(evidence['qualified_consecutive_days'], 1)
        self.assertFalse(evidence['forward_gate_met'])
        receipt = json.loads(Path(close['path']).read_text(encoding='utf-8'))
        self.assertAlmostEqual(receipt['nav'], 10099.7)

    def test_unfilled_partial_and_late_fills_do_not_qualify(self):
        empty = forward.reconcile(self.opening, self.opening, self.decision, self.close_time)
        self.assertFalse(empty['execution_aligned'])
        account = self.filled_account(quantity=50., minute='40')
        audit = forward.reconcile(self.opening, account, self.decision, self.close_time)
        self.assertFalse(audit['execution_aligned'])
        self.assertEqual(audit['minimum_order_fill_ratio'], .5)
        self.assertTrue(any('开盘' in reason for reason in audit['reasons']))

    def test_cash_injection_and_modified_old_execution_are_rejected(self):
        account = self.filled_account()
        account['cash'] += 1000.
        audit = forward.reconcile(self.opening, account, self.decision, self.close_time)
        self.assertFalse(audit['execution_aligned'])
        self.assertEqual(audit['cash_error'], 1000.)
        opening = copy.deepcopy(account)
        account['executions'].clear()
        audit = forward.reconcile(opening, account, self.decision, self.close_time)
        self.assertFalse(audit['execution_aligned'])

    def test_signal_and_close_cannot_be_backfilled(self):
        with patch.object(forward, '_now', return_value=self.close_time):
            with self.assertRaisesRegex(ValueError, '历史补记'):
                forward.freeze(self.root, self.decision, self.opening)
        self.freeze()
        with patch.object(forward, '_now', return_value=pd.Timestamp('2026-09-14 19:00', tz='Asia/Shanghai')):
            with self.assertRaisesRegex(ValueError, '历史补记'):
                forward.record_close(self.root, self.opening, self.market)

    def test_frozen_signal_and_close_are_immutable(self):
        first = self.freeze()
        self.assertEqual(first, self.freeze())
        changed = copy.deepcopy(self.decision)
        changed['orders'][0]['quantity'] = 200
        with patch.object(forward, '_now', return_value=self.signal_time):
            with self.assertRaisesRegex(ValueError, '禁止事后'):
                forward.freeze(self.root, changed, self.opening)
        account = self.filled_account()
        with patch.object(forward, '_now', return_value=self.close_time):
            forward.record_close(self.root, account, self.market)
            account['cash'] += 1
            with self.assertRaisesRegex(ValueError, '禁止覆盖'):
                forward.record_close(self.root, account, self.market)

    def test_tampered_nav_fails_independent_recalculation(self):
        self.freeze()
        account = self.filled_account()
        with patch.object(forward, '_now', return_value=self.close_time):
            close = forward.record_close(self.root, account, self.market)
            path = Path(close['path'])
            receipt = json.loads(path.read_text(encoding='utf-8'))
            receipt['nav'] += 1000
            write_json_atomic(path, receipt)
            evidence = forward.evidence(self.root)
        self.assertEqual(evidence['qualified_consecutive_days'], 0)
        self.assertEqual(len(evidence['invalid_days']), 1)

    def test_stale_evidence_is_not_current_and_no_order_day_needs_no_fake_fills(self):
        self.observation['paper_orders'] = []
        self.observation['target_weights'] = {}
        self.decision = forward.paper_decision(self.observation, self.opening, self.account_path)
        self.freeze()
        with patch.object(forward, '_now', return_value=self.close_time):
            close = forward.record_close(self.root, self.opening, self.market)
            self.assertTrue(close['qualified'])
            self.assertTrue(forward.evidence(self.root)['fresh'])
        with patch.object(forward, '_now', return_value=pd.Timestamp('2026-09-14 19:00', tz='Asia/Shanghai')):
            self.assertFalse(forward.evidence(self.root)['fresh'])

    def test_cash_dividend_receipt_reconciles_without_becoming_external_deposit(self):
        self.observation['paper_orders'] = []
        self.decision = forward.paper_decision(self.observation, self.opening, self.account_path)
        account = copy.deepcopy(self.opening)
        event = {'event_id': 'test-dividend', 'type': 'cash_dividend', 'symbol': 'SH515880',
                 'effective_date': '2026-09-11', 'cash_amount': 10., 'source': 'unit-test-confirmed-receipt'}
        account['corporate_actions'] = {'test-dividend': {'event': event,
            'fingerprint': hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()}}
        account['cash'] += 10.
        result = forward.reconcile(self.opening, account, self.decision, self.close_time)
        self.assertTrue(result['execution_aligned'])
        self.assertEqual(result['new_corporate_action_count'], 1)

    def test_126_day_gate_requires_continuity_and_resets_after_missing_session(self):
        calendar = xcals.get_calendar('XSHG')
        dates = calendar.sessions_in_range('2025-01-01', '2026-09-11')[-126:]
        self.observation['paper_orders'] = []
        self.observation['target_weights'] = {}
        for i, day in enumerate(dates):
            signal_day = calendar.previous_session(day)
            signal_time = pd.Timestamp(str(signal_day.date()) + ' 19:00', tz='Asia/Shanghai')
            close_time = pd.Timestamp(str(day.date()) + ' 19:00', tz='Asia/Shanghai')
            self.observation.update({'observation_id': f'synthetic-test-{i}', 'signal_date': str(signal_day.date()),
                'execution_date': str(day.date()), 'checked_at': signal_time.isoformat()})
            decision = forward.paper_decision(self.observation, self.opening, self.account_path)
            with patch.object(forward, '_now', return_value=signal_time):
                forward.freeze(self.root, decision, self.opening)
            self.market.calendar = pd.DatetimeIndex([day])
            self.market.bars['date'] = day
            with patch.object(forward, '_now', return_value=close_time):
                forward.record_close(self.root, self.opening, self.market)
        with patch.object(forward, '_now', return_value=self.close_time):
            evidence = forward.evidence(self.root)
            self.assertEqual(evidence['qualified_consecutive_days'], 126)
            self.assertTrue(evidence['forward_gate_met'])
            # 文件只在自动回收的测试临时目录内，不能作为真实前瞻证据。
            (self.root / 'closes' / f'{dates[99].date()}.json').unlink()
            evidence = forward.evidence(self.root)
            self.assertEqual(evidence['qualified_consecutive_days'], 26)
            self.assertFalse(evidence['forward_gate_met'])


if __name__ == '__main__':
    unittest.main()
