"""总收益与现金回放分离，以及真实行动语义、重启和失败边界测试。"""
import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.primary_cash_ledger import EventCalendar, entitlement, freeze_next, new_state, seal, settle_day, shares_after_split
from research.goal30.primary_research_series import rebuild_series
from research.goal30.reconcile_primary_execution import reconcile


class PrimaryExecutionTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(SELECTED.read_text(encoding='utf-8'))['policy']
        self.days = pd.bdate_range('2025-01-02', periods=14)
        self.symbol = 'SZ159920'

    def dividend(self, same_day=True):
        return {'event_id': 'dividend', 'type': 'cash_dividend', 'symbol': self.symbol,
            'cash_per_share_decimal': '0.2', 'record_date': str(self.days[5 if same_day else 4].date()),
            'ex_date': str(self.days[5].date()), 'announced_payment_date': str(self.days[7].date()),
            'payment_semantics': 'custodian_outgoing_transfer_actual_broker_receipt_unconfirmed',
            'entitlement_basis': 'holdings_before_record_day_trades' if same_day else 'record_date_close_holdings'}

    def context(self, i, split=False):
        price = (10. / .36555 if split else 9.8) if i >= 5 else 10.
        previous = (10. / .36555 if split else 9.8) if i >= 6 else 10.
        return {'date': str(self.days[i].date()), 'previous_date': str(self.days[i-1].date()),
                'open': {self.symbol: price}, 'close': {self.symbol: price},
                'previous_close': {self.symbol: previous}, 'volume': {self.symbol: 1e8}}

    def drive(self, event, *, confirmed=False, restore=True, buy_after=False, split=False):
        catalog = EventCalendar([event], self.days,
            payment_mode='confirmed_only' if confirmed else 'announced_schedule_model')
        state = new_state(100000., self.days[0])
        rows = []
        for i in range(1, len(self.days)):
            state, output = settle_day(state, self.context(i, split), self.policy, catalog)
            rows.append((copy.deepcopy(state), copy.deepcopy(output)))
            if i + 1 < len(self.days):
                want = i >= 5 if buy_after else i < 4
                state = freeze_next(state, {self.symbol: .8} if want else {},
                    self.context(i, split)['close'], {self.symbol: 1e8}, self.policy, self.days[i+1])
            if restore:
                state = json.loads(json.dumps(state, allow_nan=False))
        return rows

    def test_signal_total_return_accounts_for_split_dividend_and_future_prefix(self):
        raw = pd.DataFrame({'symbol': [self.symbol]*6, 'date': self.days[:6],
                            'open': [10, 10, 5.1, 5.2, 5.1, 5.2], 'close': [10, 10, 5.1, 5.2, 5.1, 5.2]})
        actions = [{'event_id': 'split', 'symbol': self.symbol, 'type': 'split', 'ratio_decimal': '2',
                    'book_date': str(self.days[1].date()), 'effective_date': str(self.days[2].date())},
                   {'event_id': 'dividend', 'symbol': self.symbol, 'type': 'cash_dividend',
                    'cash_per_share_decimal': '0.1', 'ex_date': str(self.days[4].date())}]
        series, applied = rebuild_series(raw, actions)
        np.testing.assert_allclose(series.signal_total_return_index.iloc[:5], [100, 100, 102, 104, 104])
        self.assertEqual(set(applied), {'split', 'dividend'})
        prefix, used = rebuild_series(raw.iloc[:4], actions)
        pd.testing.assert_frame_equal(prefix, series.iloc[:4].reset_index(drop=True))
        self.assertEqual(used, ['split'])
        modified = raw.copy(); modified.loc[4:, ['open', 'close']] *= 2
        after, _ = rebuild_series(modified, actions)
        pd.testing.assert_frame_equal(after.iloc[:4], series.iloc[:4])
        missing = raw.drop(index=2)
        with self.assertRaisesRegex(ValueError, '实际行情'):
            rebuild_series(missing, actions)
        with self.assertRaisesRegex(ValueError, '同日多项'):
            rebuild_series(raw, actions + [actions[0]])

    def test_exact_rounding_and_unsupported_fractional_allocation(self):
        self.assertEqual(shares_after_split(1000, {'ratio_decimal': '0.36555', 'rounding': 'ceiling_to_integer_manager_funded'}), (366, '365.55000', '0.45000'))
        self.assertEqual(shares_after_split(1000, {'ratio_decimal': '0.36555', 'rounding': 'truncate_to_integer_remainder_to_fund'})[0], 365)
        for ratio, rule, q in [('0.36555', 'registrar_exact_allocation_integer_ratio', 1000),
                               ('2', 'unknown', 1000), ('2', 'ceiling_to_integer_manager_funded', 1.5)]:
            with self.subTest(rule=rule, quantity=q), self.assertRaises(ValueError):
                shares_after_split(q, {'ratio_decimal': ratio, 'rounding': rule})

    def test_same_day_entitlement_survives_sale_and_new_buyer_has_none(self):
        rows = self.drive(self.dividend())
        opening_quantity = rows[3][0]['quantities'][self.symbol]  # 第4日收盘持仓
        ex_state, ex_output = rows[4]
        self.assertAlmostEqual(ex_state['entitled']['dividend'], opening_quantity * .2)
        records = [e for e in ex_output['actions'] if e['type'] == 'record_entitlement']
        self.assertEqual(records[0]['phase'], 'pre_open')
        self.assertEqual(ex_output['trades'], [])
        self.assertGreater(ex_output['issues']['corporate_action_order_cancelled'], 0)
        paid = [(s, o) for s, o in rows if any(e['type'] == 'pay_dividend' for e in o['actions'])]
        self.assertEqual(len(paid), 1)
        self.assertEqual(paid[0][1]['report']['date'], str(self.days[8].date()))
        self.assertNotIn(self.symbol, paid[0][0]['quantities'])
        self.assertAlmostEqual(paid[0][1]['actions'][-1]['amount'], opening_quantity * .2)
        buyer = self.drive(self.dividend(), buy_after=True)
        self.assertTrue(buyer[-1][0]['quantities'])
        self.assertEqual(buyer[-1][0]['entitled']['dividend'], 0.)

    def test_same_day_rule_does_not_recompute_from_post_trade_holdings(self):
        state = new_state(1000., self.days[4])
        state['quantities'][self.symbol] = 1000
        event = self.dividend()
        result = entitlement(state, event, 'pre_open', str(self.days[5].date()))
        state['quantities'][self.symbol] = 2000  # 当日买入不改变已记录权益。
        self.assertEqual(state['entitled']['dividend'], 200.)
        state['quantities'] = {}  # 当日卖出也不撤销权益。
        self.assertEqual(state['entitled']['dividend'], result['amount'])

    def test_confirmed_mode_retains_claim_until_explicit_receipt_and_rejects_duplicate(self):
        rows = self.drive(self.dividend(), confirmed=True)
        state = rows[7][0]  # 公告支付日之后仍未自动变成现金。
        amount = state['receivable']['dividend']
        self.assertGreater(amount, 0)
        self.assertFalse(any(e['type'] == 'pay_dividend' for _, out in rows for e in out['actions']))
        catalog = EventCalendar([self.dividend()], self.days)
        receipt = {'date': str(self.days[9].date()), 'event_id': 'dividend', 'amount': amount,
                   'receipt_id': 'synthetic-credit', 'evidence_reference': 'unit_test_synthetic_receipt'}
        paid, output = settle_day(state, self.context(9), self.policy, catalog, receipts=[receipt])
        self.assertAlmostEqual(paid['cash'] - state['cash'], amount)
        self.assertAlmostEqual(paid['previous_nav'], state['previous_nav'])
        self.assertEqual(paid['receivable'], {})
        duplicate = {**receipt, 'date': str(self.days[10].date())}
        with self.assertRaisesRegex(ValueError, '重复到账'):
            settle_day(paid, self.context(10), self.policy, catalog, receipts=[duplicate])
        wrong = {**receipt, 'amount': amount + 1}
        snapshot = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, '到账金额'):
            settle_day(state, self.context(9), self.policy, catalog, receipts=[wrong])
        self.assertEqual(state, snapshot)

    def test_split_registration_rounded_holdings_shared_budget_and_json_restart(self):
        event = {'event_id': 'split', 'symbol': self.symbol, 'type': 'split', 'ratio_decimal': '.36555',
            'record_date': str(self.days[4].date()), 'book_date': str(self.days[4].date()),
            'effective_date': str(self.days[5].date()), 'rounding': 'ceiling_to_integer_manager_funded'}
        recovered = self.drive(event, split=True)
        self.assertEqual(recovered, self.drive(event, split=True, restore=False))
        before = recovered[3][0]['quantities'][self.symbol]
        after = recovered[4][0]['quantities'][self.symbol]
        self.assertEqual(after, shares_after_split(before, event)[0])
        self.assertTrue(all(o['report']['budget_turnover'] <= .3 + 1e-12 for _, o in recovered))
        self.assertEqual(recovered[-1][0]['quantities'], {})
        self.assertTrue(any(t['side'] == 'sell' and t['quantity'] < 100 for _, o in recovered for t in o['trades']))

    def test_no_unconfirmed_cash_as_buying_power_and_no_live_model_payment(self):
        state = new_state(10., self.days[0])
        state.update(receivable={'x': 100000.}, previous_nav=100010.)
        frozen = freeze_next(seal(state), {self.symbol: .8}, {self.symbol: 10.}, {self.symbol: 1e8}, self.policy, self.days[1])
        self.assertEqual(frozen['pending_orders'], [])
        live = new_state(100000., self.days[0], mode='paper_model')
        catalog = EventCalendar([self.dividend()], self.days, payment_mode='announced_schedule_model')
        with self.assertRaisesRegex(ValueError, '只用于历史'):
            settle_day(live, self.context(1), self.policy, catalog)

    def test_payment_after_market_cutoff_remains_receivable(self):
        event = self.dividend()
        event['announced_payment_date'] = str((self.days[-1] + pd.Timedelta(days=7)).date())
        rows = self.drive(event)
        final = rows[-1][0]
        self.assertGreater(final['receivable']['dividend'], 0)
        self.assertEqual(final['receivable']['dividend'], final['entitled']['dividend'])
        self.assertFalse(any(e['type'] == 'pay_dividend' for _, out in rows for e in out['actions']))
        from research.goal30.validate_primary_execution import verify_event_exposure
        checked = {'nonzero_dividend_rights': final['entitled'], 'payments': {}}
        scenario = {'payment_delay_sessions': 0, 'uncertain_payment_extra_sessions': 1}
        self.assertEqual(verify_event_exposure(event, checked, {'final_state': final}, scenario, self.days),
                         'nonzero_receivable_payment_beyond_observation')
        with self.assertRaisesRegex(ValueError, '不得提前'):
            verify_event_exposure(event, {**checked, 'payments': final['entitled']}, {'final_state': final}, scenario, self.days)

    def test_rejects_same_day_without_semantics_gap_mutation_and_changed_payment_model(self):
        wrong = self.dividend(); wrong['entitlement_basis'] = 'record_date_close_holdings'
        with self.assertRaisesRegex(ValueError, '同日分红'):
            EventCalendar([wrong], self.days)
        catalog = EventCalendar([self.dividend()], self.days)
        state, _ = settle_day(new_state(100000., self.days[0]), self.context(1), self.policy, catalog)
        with self.assertRaisesRegex(ValueError, '衔接'):
            settle_day(state, self.context(3), self.policy, catalog)
        changed = EventCalendar([self.dividend()], self.days, uncertain_payment_extra_sessions=2)
        with self.assertRaisesRegex(ValueError, '支付模型'):
            settle_day(state, self.context(2), self.policy, changed)
        state['cash'] += 1
        with self.assertRaisesRegex(ValueError, '指纹'):
            settle_day(state, self.context(2), self.policy, catalog)

    def test_independent_reconstruction_rejects_cash_and_phase_tampering(self):
        rows = self.drive(self.dividend())
        frames = SimpleNamespace(days=self.days, context=lambda day: self.context(self.days.get_loc(pd.Timestamp(day))),
                                 adv=pd.DataFrame({self.symbol: 1e8}, index=self.days))
        report = pd.DataFrame([o['report'] for _, o in rows])
        trades = pd.DataFrame([t for _, o in rows for t in o['trades']])
        logs = pd.DataFrame([e for _, o in rows for e in o['actions']])
        holdings = pd.DataFrame([h for _, o in rows for h in o['holdings']])
        policy = {**self.policy, 'initial_capital': 100000.}
        scenario = {'payment_delay_sessions': 0, 'uncertain_payment_extra_sessions': 1, 'slippage_multiplier': 1.}
        result = reconcile(frames, [self.dividend()], policy, scenario, report, trades, logs, holdings, rows[-1][0])
        self.assertLess(result['maximum_absolute_numeric_error'], 1e-7)
        bad = report.copy(); bad.loc[0, 'cash'] += .01
        with self.assertRaisesRegex(ValueError, '重建不符'):
            reconcile(frames, [self.dividend()], policy, scenario, bad, trades, logs, holdings, rows[-1][0])
        bad = logs.copy(); bad.loc[bad.type.eq('record_entitlement'), 'phase'] = 'post_close'
        with self.assertRaisesRegex(ValueError, '同日权益'):
            reconcile(frames, [self.dividend()], policy, scenario, report, trades, bad, holdings, rows[-1][0])


if __name__ == '__main__':
    unittest.main()
