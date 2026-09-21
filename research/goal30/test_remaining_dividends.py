"""原件提取的身份、单位、时间和领取歧义回归测试。"""
import unittest

from research.goal30.remaining_dividend_evidence import compact, parse_notice, extract_sources
from research.goal30.acquire_remaining_dividends import ROOT


class RemainingDividendTests(unittest.TestCase):
    def setUp(self):
        self.text = compact((ROOT/'SH510050_2024-12-02_notice_1.txt').read_text(encoding='utf-8'))
        self.conflict = compact((ROOT/'SH510880_2017-01-23_supplement_1.txt').read_text(encoding='utf-8'))

    def parse(self, text):
        return parse_notice([text], 'SH510050')

    def test_identity_and_amount_units(self):
        for old, new in [('基金主代码510050','基金主代码510300'),
                         ('华夏基金管理有限公司','另一基金管理有限公司'),
                         ('元/10份基金份额','元/100份基金份额'),
                         (')0.55',')0.00'), (')0.55',')-0.55')]:
            with self.subTest(old=old,new=new):
                self.assertIn(old,self.text)
                with self.assertRaises(ValueError): self.parse(self.text.replace(old,new))

    def test_date_order_and_competing_table_rejected(self):
        for text in [self.text.replace('权益登记日2024年11月29日','权益登记日2024年12月3日'),
                     self.text.replace('现金红利发放日2024年12月5日','现金红利发放日2024年12月1日'),
                     self.text+'权益登记日2024年11月29日除息日2024年12月2日现金红利发放日2024年12月6日分红对象']:
            with self.subTest(text=text[-90:]), self.assertRaises(ValueError): self.parse(text)

    def test_venue_ambiguity_rejected(self):
        with self.assertRaisesRegex(ValueError,'场外'): self.parse(self.text+'场外份额')

    def test_designated_receipt_and_custody_required(self):
        for old, new in [('领取现金红利','领取其他款项'),('未办理指定交易','待办交易'),
                         ('保管','处理'),('已在证券公司处办理指定交易的投资者','部分投资者'),
                         ('第二个交易日可领取现金红利','将来可领取现金红利')]:
            with self.subTest(old=old):
                self.assertIn(old,self.text)
                with self.assertRaises(ValueError): self.parse(self.text.replace(old,new))

    def test_generic_and_explicit_same_date_claim(self):
        for phrase in ['现金红利发放日','红利发放日','2024年12月5日']:
            row=self.parse(self.text.replace('投资者可在现金红利发放日','投资者可在'+phrase))
            self.assertTrue(row['payment_semantics_verified'])
            self.assertFalse(row['event']['actual_receipt_confirmed'])
            self.assertIsNone(row['event']['actual_cash_available_date'])

    def test_original_conflicting_year_is_retained(self):
        row=parse_notice([self.conflict],'SH510880')
        self.assertEqual(row['event']['announced_payment_date'],'2017-01-26')
        self.assertEqual(row['internal_investor_claim_date_conflicts'],['2016-01-26'])
        self.assertFalse(row['payment_semantics_verified'])
        self.assertEqual(row['event']['payment_semantics'],'announced_distribution_date_actual_broker_receipt_unconfirmed')
        corrected=parse_notice([self.conflict.replace('投资者可在2016年1月26日','投资者可在2017年1月26日')],'SH510880')
        self.assertTrue(corrected['payment_semantics_verified'])

    def test_extra_dash_only_in_visually_reviewed_template(self):
        self.assertIn('除息日2017年1月23日-现金红利发放日',self.conflict)
        parse_notice([self.conflict],'SH510880')
        with self.assertRaises(ValueError):
            parse_notice([self.conflict.replace('公告送出日期:2017年1月17日','公告送出日期:2017年1月18日')],'SH510880')
        with self.assertRaises(ValueError):
            self.parse(self.text.replace('除息日2024年12月2日','除息日2024年12月2日-'))

    def test_all_original_templates_and_cross_page_clause(self):
        result=extract_sources()
        self.assertEqual(len(result['events']),33)
        self.assertEqual(sum(r['payment_semantics_verified'] for r in result['events']),32)
        self.assertEqual(result['relevant_pages'],66)


if __name__=='__main__': unittest.main(verbosity=2)
