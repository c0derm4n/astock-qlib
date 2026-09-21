import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.goal30.action_sensitivity import adjusted_split_bars
from research.goal30.data_evidence import compare_events, parse_eastmoney, parse_sina
from research.goal30 import data_evidence


class ActionEvidenceTests(unittest.TestCase):
    def test_dividend_units_and_split_direction(self):
        html = '''<title>测试(515880)</title><table><tr><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr>
        <tr><td>2026-01-16</td><td>2026-01-19</td><td>每10份派现金1.2300元</td><td>2026-01-27</td></tr></table>
        <table><tr><th>拆分折算日</th><th>拆分折算比例</th></tr><tr><td>2026-02-02</td><td>1:3.0000</td></tr></table>'''
        events = parse_eastmoney(html, '515880')
        self.assertAlmostEqual(events[0]['cash_per_share'], .123)
        self.assertEqual(events[0]['payment_date'], '2026-01-27')
        self.assertEqual(events[1]['ratio'], 3.)
        with self.assertRaises(ValueError):
            parse_eastmoney(html, '518880')

    def test_sina_is_parsed_as_data_and_unknown_script_tail_rejected(self):
        text = 'var sh515880hfq={"data":[{"d":"1900-01-01","s":1,"u":0},{"d":"2026-02-03","s":3,"u":0}]};'
        self.assertEqual(parse_sina(text, 'SH515880')[0]['ratio'], 3.)
        with self.assertRaises(ValueError):
            parse_sina(text + 'alert(1)', 'SH515880')

    def test_date_difference_and_rounding_conflict_are_not_marked_audited(self):
        em = [{'type': 'split', 'book_date': '2026-02-02', 'ratio': 3.}]
        sina = [{'type': 'split', 'effective_date': '2026-02-03', 'ratio': 3.}]
        result = compare_events(em, sina)
        self.assertEqual(result['unresolved_date_count'], 1)
        self.assertFalse(result['issuer_audit_complete'])
        sina[0]['ratio'] = 3.001
        result = compare_events(em, sina)
        self.assertEqual(len(result['unmatched']), 2)

    def test_split_sensitivity_changes_only_instrument_and_dates_after_event(self):
        bars = pd.DataFrame({'symbol': ['SH515880'] * 3 + ['SH518880'],
            'date': pd.to_datetime(['2026-02-02', '2026-02-03', '2026-02-04', '2026-02-03']),
            'open': [30., 10., 11., 100.], 'close': [30., 10., 11., 100.], 'adj_factor': [1., 3.001, 3.001, 1.]})
        corrected, _ = adjusted_split_bars(bars, 'SH515880', [{'effective_date': '2026-02-03', 'ratio': 3.}])
        self.assertEqual(bars.adj_factor.iloc[1], 3.001)
        self.assertEqual(corrected.adj_factor.iloc[0], 1.)
        self.assertEqual(corrected.adj_factor.iloc[3], 1.)
        self.assertAlmostEqual(corrected.adj_close.iloc[0], corrected.adj_close.iloc[1])
        self.assertAlmostEqual(corrected.adj_close.iloc[2], 33.)

    def test_rate_limit_stops_remaining_requests(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(data_evidence, 'OUT', Path(temp)), \
                patch.object(data_evidence, 'SYMBOLS', {'SH510300', 'SH515880'}), \
                patch.object(data_evidence, 'fetch', side_effect=data_evidence.SourceRateLimit('limited')) as fetch, \
                patch('builtins.print'):
            directory = data_evidence.collect()
            result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(result['symbols']['SH515880']['error_type'], 'NotRequestedAfterRateLimit')
            self.assertFalse(result['corporate_actions_audited'])


if __name__ == '__main__':
    unittest.main()
