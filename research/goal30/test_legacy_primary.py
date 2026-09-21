"""原件财务字段和覆盖边界的故障验证；不写入或修改任何证据文件。"""
import unittest

from research.goal30.verify_legacy_primary import LEGACY, merge_coverage, parse_gold_2018, parse_notice


class PrimaryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pypdf import PdfReader
        cls.gold = [p.extract_text() for p in PdfReader(LEGACY / '518880_2018_annual.pdf').pages]
        cls.notice = '\n'.join(p.extract_text() for p in PdfReader(LEGACY / '510300_dividend_2019-12-05.pdf').pages)

    def test_investor_payment_date_and_per_share_units(self):
        event, _ = parse_notice(self.notice)
        self.assertEqual(event['payment_date'], '2019-12-16')
        self.assertEqual(event['cash_per_share'], 0.062)
        self.assertNotEqual(event['payment_date'], '2019-12-13')  # 券商划款日不可作到账日。
        for old, new in [('510300', '510500'), ('元/10', '元/100'), ('现金红利发放日', '划款日')]:
            with self.subTest(field=old), self.assertRaises(ValueError):
                parse_notice(self.notice.replace(old, new))

    def test_gold_financial_columns_and_faults(self):
        result = parse_gold_2018(self.gold)
        self.assertEqual(result['share_movement_evidence']['end_shares'], '2636740847.00')
        self.assertEqual(result['share_movement_evidence']['unexplained_net_share_change'], '0.00')
        self.assertTrue(result['cash_distribution_permitted'])
        faults = [(4, '518880', '518800'), (8, '没有利润分配', '进行了利润分配'),
                  (27, '采用现金方式', '不采用现金方式'),
                  (34, '-2,039,400,000.00', '2,039,400,000.00'),
                  (34, '2,636,740,847.00', '2,636,740,848.00'),
                  (34, '基金份额（份） 账面金额', '账面金额 基金份额（份）')]
        for index, old, new in faults:
            pages = self.gold.copy()
            self.assertIn(old, pages[index])
            pages[index] = pages[index].replace(old, new)
            with self.subTest(field=old), self.assertRaises(ValueError):
                parse_gold_2018(pages)

    def test_coverage_does_not_fill_missing_days(self):
        periods = [('2016-01-01', '2018-12-31'), ('2019-01-01', '2021-12-31'),
                   ('2020-01-01', '2022-12-31'), ('2023-01-01', '2025-12-31'),
                   ('2026-01-01', '2026-06-30')]
        self.assertEqual(merge_coverage(periods), [['2016-01-01', '2026-06-30']])
        self.assertEqual(merge_coverage([periods[0], periods[2]]),
                         [['2016-01-01', '2018-12-31'], ['2020-01-01', '2022-12-31']])
        with self.assertRaises(ValueError):
            merge_coverage([('2026-06-30', '2026-01-01')])


if __name__ == '__main__':
    unittest.main()
