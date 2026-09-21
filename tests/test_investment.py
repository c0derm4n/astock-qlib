import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import config
from src.runs import read_protocol
from src.portfolio import target_weights, plan_orders, buy_notional_budget
from src.account import initialize, apply_fills
from src.datasource.free_source import aligned_increment, fetch_actions
from src.datasource import DataStore
from src.free_data import run_free_update


class InvestmentTests(unittest.TestCase):
    def setUp(self):
        self.policy = read_protocol()
        self.symbols = ['SH510300', 'SH510050', 'SH510500', 'SH510880', 'SH511260', 'SH518880']
        self.returns = pd.DataFrame(np.random.default_rng(2).normal(0, .005, (80, 6)), columns=self.symbols)

    def test_boundary_ties_do_not_select_by_symbol(self):
        policy = dict(self.policy, portfolio_style='rank', topk=2)
        scores = pd.Series([2., 1., 1., 1.], index=self.symbols[:4])
        target = target_weights(scores, scores.index, {}, {}, self.returns, policy)
        self.assertEqual(set(target), {self.symbols[0]})
        shuffled = target_weights(scores.iloc[::-1], scores.index, {}, {}, self.returns, policy)
        self.assertEqual(target, shuffled)

    def test_pending_exit_is_not_bought_back_on_rank_recovery(self):
        scores = pd.Series([2., 1.], index=self.symbols[:2])
        current = {self.symbols[0]: .1}
        target = target_weights(scores, scores.index, current, {self.symbols[0]: 30}, self.returns,
                                self.policy, exiting=[self.symbols[0]])
        self.assertNotIn(self.symbols[0], target)

    def test_core_has_explicit_budget_and_missing_asset_stays_cash(self):
        policy = dict(self.policy, max_daily_turnover=1.)
        scores = pd.Series(np.arange(6), index=self.symbols)
        target = target_weights(scores, scores.index, {}, {}, self.returns, policy)
        self.assertAlmostEqual(target['SH511260'], .2)
        self.assertAlmostEqual(target['SH518880'], .1)
        missing = target_weights(scores, self.symbols[:-1], {}, {}, self.returns, policy)
        self.assertNotIn('SH518880', missing)
        self.assertAlmostEqual(missing['SH511260'], .2)

    def test_buy_plans_do_not_spend_expected_sale_proceeds(self):
        orders = plan_orders({'SH510050': .2}, {'SH510300': 2000}, 100,
                             {'SH510300': 4., 'SH510050': 3.}, dict.fromkeys(self.symbols, 1e9), self.policy)
        self.assertTrue(any(o['side'] == 'sell' for o in orders))
        self.assertFalse(any(o['side'] == 'buy' for o in orders))
        room = buy_notional_budget('SH515070', {'SH515880': 4000}, 60000,
                                   {'SH515070': 10., 'SH515880': 10.}, self.policy)
        self.assertEqual(room, 0)  # 同风险组已超过35%

    def test_partial_exit_persists_in_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'account.json'
            initialize(path, 1000, {'SH510300': {'quantity': 200, 'average_cost': 4., 'buy_date': '2026-07-01'}})
            decision = {'decision_id': 'd', 'account_revision': 0, 'execution_date': '2026-07-27', 'orders': [
                {'order_id': 'o', 'symbol': 'SH510300', 'side': 'sell', 'quantity': 200, 'target_weight': 0}]}
            fill = {'execution_id': 'f', 'decision_id': 'd', 'order_id': 'o', 'symbol': 'SH510300', 'side': 'sell',
                    'quantity': 100, 'price': 4., 'fee': .2, 'executed_at': '2026-07-27T09:30:00+08:00'}
            result = apply_fills(path, decision, [fill])
            self.assertTrue(result['holdings']['SH510300']['exit_pending'])


class FreeSourceTests(unittest.TestCase):
    def setUp(self):
        self.days = xcals.get_calendar('XSHG').sessions_in_range('2026-06-01', '2026-07-27')
        self.raw = pd.DataFrame({'symbol': 'SH510300', 'date': self.days, 'open': 4., 'high': 4.1,
            'low': 3.8, 'close': 4., 'volume': 1e6, 'amount': 4e8, 'pct_chg': 0.})
        self.actions = pd.DataFrame({'date': [pd.Timestamp('1900-01-01')], 'share_factor': [1.], 'cumulative_cash': [0.]})
        self.old = self.raw.iloc[:30].assign(adj_factor=1.)

    def test_cash_dividend_is_not_misclassified_as_return(self):
        raw = self.raw.copy()
        raw.loc[30:, 'close'] = 3.9
        actions = pd.concat([self.actions, pd.DataFrame({'date': [self.days[30]], 'share_factor': [1.], 'cumulative_cash': [.1]})])
        new, audit = aligned_increment(raw, actions, self.old)
        self.assertAlmostEqual(new.iloc[0]['close'] * new.iloc[0]['adj_factor'], 4.)
        self.assertEqual(audit['factor_error'], 0)

    def test_historical_split_and_rounding_are_anchored_exactly(self):
        raw = self.raw.copy()
        raw.loc[25:, 'close'] = 2.
        old = raw.iloc[:30].assign(adj_factor=1.)
        old.loc[25:, 'adj_factor'] = 1.9993
        actions = pd.concat([self.actions, pd.DataFrame({'date': [self.days[25]], 'share_factor': [2.], 'cumulative_cash': [0.]})])
        new, audit = aligned_increment(raw, actions, old)
        self.assertAlmostEqual(new.iloc[0]['adj_factor'], 1.9993)
        self.assertLess(audit['factor_error'], .003)

    def test_inconsistent_raw_or_missing_trading_day_blocks_update(self):
        bad = self.raw.copy()
        bad.loc[10, 'close'] = 8.
        with self.assertRaisesRegex(ValueError, '不一致'):
            aligned_increment(bad, self.actions, self.old)
        with self.assertRaisesRegex(ValueError, '交易日缺失'):
            aligned_increment(self.raw.drop(index=31), self.actions, self.old)

    def test_remote_script_is_parsed_without_eval(self):
        body = 'var sh510300hfq={"data":[{"d":"1900-01-01","s":"1","u":"0"}]}\n/* source checksum */'
        with patch('src.datasource.free_source._get', return_value=body):
            self.assertEqual(len(fetch_actions('SH510300')), 1)
        with patch('src.datasource.free_source._get', return_value=body + '\nalert(1)'):
            with self.assertRaisesRegex(ValueError, '未知脚本'):
                fetch_actions('SH510300')

    def test_failed_source_does_not_replace_original_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / 'market.duckdb'
            with DataStore(db) as store:
                data = self.old.copy().assign(source='tushare', fetch_time='2026-07-27', data_version='old')
                store.upsert_bars(data.drop(columns='adj_factor'))
                store.upsert_adj(data[['symbol','date','adj_factor','source','fetch_time','data_version']])
            before = db.read_bytes()
            with patch.object(config, 'DUCKDB_PATH', db), patch.object(config, 'OUTPUT_DIR', root / 'output'), \
                 patch('src.free_data.fetch_recent_bars', side_effect=ValueError('test source failure')):
                with self.assertRaisesRegex(ValueError, '原库未变'):
                    run_free_update()
            self.assertEqual(db.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
