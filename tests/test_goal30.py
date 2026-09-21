import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from research.goal30.audit import comparison, wealth_metrics
from research.goal30 import daily
from src.account import apply_fills, initialize


class DrawdownDefinitionTests(unittest.TestCase):
    def test_daily_average_includes_new_high_and_recovery_days(self):
        series = pd.Series([100., 80., 100., 110.], index=pd.date_range('2020-01-01', periods=4))
        metrics = wealth_metrics(series)
        self.assertAlmostEqual(metrics['mean_daily_drawdown'], .05)
        self.assertAlmostEqual(metrics['max_drawdown'], .2)
        self.assertIsNone(metrics['mean_annual_max_drawdown'])

    def test_annual_reset_includes_loss_from_previous_year_close(self):
        series = pd.Series([100., 50., 100., 80., 80.], index=pd.to_datetime(
            ['2019-12-31', '2020-01-02', '2020-12-31', '2021-01-04', '2022-01-04']))
        metrics = wealth_metrics(series)
        self.assertEqual(set(metrics['annual_max_drawdowns']), {'2020', '2021'})
        self.assertAlmostEqual(metrics['annual_max_drawdowns']['2020'], .5)
        self.assertAlmostEqual(metrics['annual_max_drawdowns']['2021'], .2)
        self.assertAlmostEqual(metrics['mean_annual_max_drawdown'], .35)

    def test_comparison_rejects_silently_dropped_dates(self):
        series = pd.Series([100., 80., 110.], index=pd.date_range('2020-01-01', periods=3))
        with self.assertRaisesRegex(ValueError, '相同交易日'):
            comparison(series, series.iloc[[0, 2]])

    def test_missing_year_nonfinite_and_duplicate_dates_are_rejected(self):
        for dates, values in [(['2019-12-31', '2022-01-01'], [100., 110.]),
                              (['2020-01-01', '2020-01-02'], [100., float('nan')]),
                              (['2020-01-01', '2020-01-01'], [100., 110.])]:
            with self.assertRaises(ValueError):
                wealth_metrics(pd.Series(values, index=pd.to_datetime(dates)))


class DailyObservationTests(unittest.TestCase):
    def setUp(self):
        self.selected = json.loads(daily.SELECTED.read_text(encoding='utf-8'))
        dates = pd.bdate_range(end='2026-09-11', periods=70)
        bars = pd.DataFrame([(d, s, 10., 10.) for d in dates for s in daily.SYMBOLS],
                            columns=['date', 'symbol', 'close', 'adj_close'])
        eligibility = bars[['date', 'symbol']].assign(eligible=True, avg_amount=1e9)
        self.market = SimpleNamespace(bars=bars, eligibility=eligibility, calendar=dates, fingerprint='test')
        self.state = daily.session_state(pd.Timestamp('2026-09-11 19:00', tz='Asia/Shanghai'))
        self.account = {'cash': 1e6, 'holdings': {}, 'revision': 0}

    def test_weekend_and_before_close_do_not_load_market_or_refresh(self):
        for now, status in [('2026-09-13 19:00', 'market_closed'), ('2026-09-11 18:59', 'not_due')]:
            with tempfile.TemporaryDirectory() as temp, patch.object(daily, 'load_market') as market, \
                    patch.object(daily.subprocess, 'run') as refresh:
                result = daily.run(output_dir=temp, refresh=True, now=pd.Timestamp(now, tz='Asia/Shanghai'))
                self.assertEqual(result['status'], status)
                market.assert_not_called()
                refresh.assert_not_called()

    def test_timezone_next_session_and_stale_data(self):
        self.assertEqual(self.state['execution_date'], '2026-09-14')
        self.assertEqual(daily.session_state(pd.Timestamp('2026-09-11 11:00Z'))['status'], 'due')
        self.market.calendar = self.market.calendar[:-1]
        with self.assertRaisesRegex(ValueError, '过期数据'):
            daily.observe(self.selected, self.market, self.state)

    def test_missing_account_does_not_assume_empty_portfolio(self):
        result = daily.observe(self.selected, self.market, self.state)
        self.assertIsNone(result['rebalance_needed'])
        self.assertEqual(result['orders'], [])
        self.assertEqual(result['paper_orders'], [])
        self.assertFalse(result['executable'])
        self.assertEqual(result['allocation_reference'], {'SH515880': .6, 'SH518880': .35})

    def test_eligibility_fallback_and_cash(self):
        self.market.eligibility.loc[self.market.eligibility.symbol.eq('SH515880'), 'eligible'] = False
        result = daily.observe(self.selected, self.market, self.state)
        self.assertEqual(result['allocation_reference'], {'SH510300': .6, 'SH518880': .35})
        self.market.eligibility.loc[self.market.eligibility.symbol.eq('SH510300'), 'eligible'] = False
        result = daily.observe(self.selected, self.market, self.state)
        self.assertEqual(result['allocation_reference'], {'SH518880': .35})
        self.assertAlmostEqual(result['cash_reference_weight'], .65)

    def test_paper_build_is_bounded_and_cannot_create_live_orders(self):
        result = daily.observe(self.selected, self.market, self.state, self.account)
        self.assertTrue(result['rebalance_needed'])
        self.assertLessEqual(sum(result['target_weights'].values()), .3 + 1e-10)
        self.assertTrue(all(o['quantity'] % 100 == 0 for o in result['paper_orders']))
        self.assertTrue(all(o['symbol'] in daily.SYMBOLS for o in result['paper_orders']))
        self.assertEqual(result['orders'], [])
        self.assertFalse(result['executable'])

    def test_pool_external_account_and_missing_quote_are_rejected(self):
        self.account['holdings'] = {'SH513100': {'quantity': 100, 'buy_date': '2026-09-10'}}
        with self.assertRaisesRegex(ValueError, '池外持仓'):
            daily.observe(self.selected, self.market, self.state, self.account)
        self.market.bars = self.market.bars[~self.market.bars.symbol.eq('SH518880')]
        with self.assertRaisesRegex(ValueError, '当日正式价格'):
            daily.observe(self.selected, self.market, self.state)

    def test_same_day_is_immutable_and_does_not_refresh_twice(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(daily, 'load_selected', return_value=(self.selected, Path(temp))), \
                patch.object(daily, 'load_market', return_value=self.market) as market:
            now = pd.Timestamp('2026-09-11 19:00', tz='Asia/Shanghai')
            first = daily.run(output_dir=temp, now=now)
            second = daily.run(output_dir=temp, now=now)
            self.assertEqual(first['observation_id'], second['observation_id'])
            self.assertTrue(second['already_recorded'])
            self.assertFalse(second['notification_needed'])
            self.assertEqual(market.call_count, 1)

    def test_daily_account_workflow_freezes_importable_orders_and_records_next_close(self):
        with tempfile.TemporaryDirectory() as temp:
            account_path = Path(temp) / 'account.json'
            initialize(account_path, 1e6)
            output = Path(temp) / 'observations'
            first_market = SimpleNamespace(bars=self.market.bars[self.market.bars.date.le('2026-09-10')],
                eligibility=self.market.eligibility[self.market.eligibility.date.le('2026-09-10')],
                calendar=self.market.calendar[:-1], fingerprint='first-day')
            first_time = pd.Timestamp('2026-09-10 19:00', tz='Asia/Shanghai')
            second_time = pd.Timestamp('2026-09-11 19:00', tz='Asia/Shanghai')
            with patch.object(daily, 'load_selected', return_value=(self.selected, Path(temp))), \
                    patch.object(daily, 'load_market', return_value=first_market), \
                    patch.object(daily.forward, '_now', return_value=first_time):
                first = daily.run(account_path=account_path, output_dir=output, now=first_time)
            self.assertEqual(first['status'], 'research_only', first)
            decision = json.loads(Path(first['paper_decision_path']).read_text(encoding='utf-8'))
            fills = [{'execution_id': o['order_id'] + '-fill', 'decision_id': decision['decision_id'],
                      'order_id': o['order_id'], 'symbol': o['symbol'], 'side': o['side'],
                      'quantity': o['quantity'], 'price': 10., 'fee': o['quantity'] * 10. * .0003,
                      'executed_at': '2026-09-11T09:30:00+08:00'} for o in decision['orders']]
            apply_fills(account_path, decision, fills)
            with patch.object(daily, 'load_selected', return_value=(self.selected, Path(temp))), \
                    patch.object(daily, 'load_market', return_value=self.market), \
                    patch.object(daily.forward, '_now', return_value=second_time):
                second = daily.run(account_path=account_path, output_dir=output, now=second_time)
            self.assertEqual(second['status'], 'research_only', second)
            self.assertTrue(second['forward_close']['qualified'])
            self.assertEqual(second['forward_admission_days'], 1)
            self.assertFalse(second['investment_admission'])
            self.assertEqual(second['orders'], [])


if __name__ == '__main__':
    unittest.main()
