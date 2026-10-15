"""周末持仓取实际成交后的记录，保留空仓与休市周。"""
import unittest
import pandas as pd
from src.formal_history import weekly_holdings_history


class WeeklyHoldingsTests(unittest.TestCase):
    def test_snapshot_uses_week_end_and_includes_current_week(self):
        dates = pd.to_datetime(['2026-09-28', '2026-09-29', '2026-09-30', '2026-10-08', '2026-10-09', '2026-10-12'])
        report = pd.DataFrame({'account': [1000]*6, 'cash': [500, 1000, 1000, 1000, 1000, 1000],
                               'value': [500, 0, 0, 0, 0, 0]}, index=dates)
        holdings = pd.DataFrame([{'date': dates[0], 'symbol': 'SH515880', 'quantity': 100, 'value': 500}])
        result = weekly_holdings_history(report, holdings)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['date'], '2026-10-12')
        self.assertTrue(result[0]['partial'])
        self.assertEqual(result[1]['date'], '2026-10-09')
        self.assertFalse(result[1]['partial'])
        self.assertEqual(result[2]['date'], '2026-09-30')
        self.assertTrue(all(r['empty'] and r['cash_weight'] == 1 for r in result))

    def test_full_holiday_week_carries_last_valid_holdings(self):
        dates = pd.to_datetime(['2024-02-08', '2024-02-19'])
        report = pd.DataFrame({'account': [1000, 1000], 'cash': [500, 1000], 'value': [500, 0]}, index=dates)
        holdings = pd.DataFrame([{'date': dates[0], 'symbol': 'SH510300', 'quantity': 100, 'value': 500}])
        result = weekly_holdings_history(report, holdings)
        holiday = next(r for r in result if r['week_start'] == '2024-02-12')
        self.assertTrue(holiday['carried_over'])
        self.assertEqual(holiday['date'], '2024-02-08')
        self.assertEqual(holiday['holdings'][0]['quantity'], 100)
        self.assertFalse(result[0]['holdings'])

    def test_receivables_are_not_available_cash_and_weights_reconcile(self):
        date = pd.Timestamp('2026-10-12')
        report = pd.DataFrame({'account': [1000], 'cash': [400], 'value': [500], 'dividend_receivable': [100]}, index=[date])
        holdings = pd.DataFrame([{'date': date, 'symbol': 'SH510300', 'quantity': 100, 'value': 500}])
        row = weekly_holdings_history(report, holdings)[0]
        self.assertEqual(row['cash_weight'], .4)
        self.assertEqual(row['receivable_weight'], .1)
        self.assertAlmostEqual(row['cash_weight'] + row['receivable_weight'] + row['exposure'], 1)

    def test_missing_or_duplicate_holdings_fail_instead_of_appearing_empty(self):
        date = pd.Timestamp('2026-10-12')
        report = pd.DataFrame({'account': [1000], 'cash': [500], 'value': [500]}, index=[date])
        with self.assertRaises(ValueError):
            weekly_holdings_history(report, pd.DataFrame())
        holding = {'date': date, 'symbol': 'SH510300', 'quantity': 100, 'value': 500}
        with self.assertRaises(ValueError):
            weekly_holdings_history(report, pd.DataFrame([holding, holding]))

    def test_missing_trading_week_cannot_be_labelled_holiday(self):
        report = pd.DataFrame({'account': [1000, 1000], 'cash': [1000, 1000], 'value': [0, 0]},
                              index=pd.to_datetime(['2026-09-18', '2026-10-12']))
        with self.assertRaises(ValueError):
            weekly_holdings_history(report, pd.DataFrame())
