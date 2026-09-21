import json
import unittest
from types import SimpleNamespace

import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.sp500.explore import run_candidate


class CashDividendReplayTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(SELECTED.read_text(encoding='utf-8'))['policy']
        self.policy.update(initial_capital=10000., max_daily_turnover=1., commission=0.,
                           minimum_commission=0., slippage=0., impact_coefficient=0.)
        self.candidate = {'name': 'unit-test', 'family': 'fixed', 'weights': {'SH510300': .5}}
        self.dates = pd.to_datetime(['2026-01-09', '2026-01-12', '2026-01-13', '2026-01-14',
                                    '2026-01-15', '2026-01-16', '2026-01-19'])
        self.event = {'event_id': 'test-dividend', 'type': 'cash_dividend', 'symbol': 'SH510300',
                      'record_date': '2026-01-13', 'ex_date': '2026-01-14',
                      'payment_date': '2026-01-16', 'cash_per_share': 1.}

    def market(self, prices, factors=None):
        bars = pd.DataFrame({'date': self.dates, 'symbol': 'SH510300', 'open': prices,
                             'close': prices, 'volume': 1e6, 'adj_factor': factors or [1.] * len(prices)})
        bars['adj_open'] = bars.open * bars.adj_factor
        bars['adj_close'] = bars.close * bars.adj_factor
        market = SimpleNamespace(bars=bars, calendar=self.dates,
            eligibility=bars[['date', 'symbol']].assign(eligible=True, avg_amount=1e9))
        market.wide = lambda field: market.bars.pivot(index='date', columns='symbol', values=field)
        return market

    def run_cash(self, market, actions, **kwargs):
        report, trades, log, holdings, issues = simulate_cash(market, self.candidate, self.policy,
                                                             '2026-01-12', actions, **kwargs)
        audit = audit_cash_ledger(market, report, trades, log, 10000.)
        self.assertEqual(audit['violations'], [])
        return report, trades, log, holdings

    def test_no_action_case_matches_original_event_engine(self):
        market = self.market([10., 10., 10., 10.1, 10.2, 10.3, 10.4])
        report, trades, _, _ = self.run_cash(market, [])
        scores = pd.DataFrame(0., index=self.dates, columns=['SH510300'])
        original, original_trades, _ = run_candidate(market, scores, self.candidate, self.policy, '2026-01-12')
        pd.testing.assert_frame_equal(report.drop(columns='dividend_receivable'), original)
        pd.testing.assert_frame_equal(trades, original_trades)

    def test_ex_date_receivable_preserves_nav_and_payment_is_not_prespent(self):
        market = self.market([10., 10., 10., 9., 9., 9., 9.], [1., 1., 1., 10/9, 10/9, 10/9, 10/9])
        report, trades, log, _ = self.run_cash(market, [self.event])
        self.assertEqual(report.loc['2026-01-14', 'dividend_receivable'], 500.)
        self.assertEqual(report.loc['2026-01-14', 'cash'], 5000.)
        self.assertEqual(report.loc['2026-01-14', 'account'], 10000.)
        self.assertEqual(report.loc['2026-01-16', 'dividend_receivable'], 0.)
        self.assertEqual(report.loc['2026-01-16', 'cash'], 5500.)
        self.assertEqual(list(log[log.type.eq('pay_dividend')].date), ['2026-01-16'])
        self.assertFalse(trades.date.eq('2026-01-16').any())

    def test_payment_delay_keeps_claim_unavailable_and_sale_preserves_entitlement(self):
        market = self.market([10., 10., 10., 9., 9., 9., 9.], [1., 1., 1., 10/9, 10/9, 10/9, 10/9])
        market.eligibility.loc[market.eligibility.date.ge('2026-01-13'), 'eligible'] = False
        report, trades, log, _ = self.run_cash(market, [self.event], payment_delay_sessions=1)
        self.assertTrue(((trades.side == 'sell') & (trades.date == '2026-01-15')).any())
        self.assertEqual(report.loc['2026-01-16', 'dividend_receivable'], 500.)
        self.assertEqual(report.loc['2026-01-19', 'cash'], 10000.)
        self.assertEqual(log.loc[log.type.eq('pay_dividend'), 'amount'].iloc[0], 500.)

    def test_post_record_purchase_does_not_receive_dividend(self):
        market = self.market([10., 10., 10., 9., 9., 9., 9.])
        market.eligibility.loc[market.eligibility.date.lt('2026-01-14'), 'eligible'] = False
        report, trades, log, _ = self.run_cash(market, [self.event])
        self.assertTrue((trades.date == '2026-01-15').any())
        self.assertEqual(log.loc[log.type.eq('pay_dividend'), 'amount'].iloc[0], 0.)
        self.assertTrue(report.dividend_receivable.eq(0.).all())

    def test_split_changes_shares_without_creating_cash_or_return(self):
        market = self.market([30., 30., 30., 10., 10., 10., 10.], [1., 1., 1., 3., 3., 3., 3.])
        self.candidate['weights']['SH510300'] = .6
        action = {'event_id': 'test-split', 'symbol': 'SH510300', 'type': 'split',
                  'effective_date': '2026-01-14', 'ratio': 3.}
        report, _, _, holdings = self.run_cash(market, [action])
        quantity = holdings.set_index('date').quantity
        self.assertEqual(quantity.loc['2026-01-13'], 200.)
        self.assertEqual(quantity.loc['2026-01-14'], 600.)
        self.assertTrue(report.account.eq(10000.).all())


if __name__ == '__main__':
    unittest.main()
