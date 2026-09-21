"""防止把场外日期或基金划款日误当场内投资者现金到账。"""
import unittest

from research.goal30.bond_dividend_evidence import ROOT,parse_bond_notice
from research.goal30.broad_data_readiness import in_window


class BondEvidenceTests(unittest.TestCase):
    def text(self,day):
        return (ROOT/f'511260_dividend_{day}.txt').read_text(encoding='utf-8')

    def test_four_originals_select_exchange_traded_dates_and_per_share_units(self):
        expected={
            '2025-09-18':('2025-09-22','2025-09-23','2025-09-26',1.36),
            '2025-12-23':('2025-12-25','2025-12-26','2025-12-31',.833),
            '2026-03-20':('2026-03-24','2026-03-25','2026-03-30',.6711),
            '2026-06-22':('2026-06-24','2026-06-25','2026-06-30',1.2686)}
        for day,values in expected.items():
            with self.subTest(day=day):
                event=parse_bond_notice(self.text(day))['event']
                self.assertEqual(tuple(event[k] for k in ('record_date','ex_date','payment_date','cash_per_share')),values)
                self.assertEqual(event['share_class'],'exchange_traded')

    def test_ambiguous_venue_wrong_identity_or_unit_fail(self):
        text=self.text('2026-06-22')
        for broken in [text.replace('511260','511261'),text.replace('场内','场外'),
                       text.replace('元/10','元/100'),text.replace('国泰基金管理有限公司','其他管理人')]:
            with self.subTest(),self.assertRaises(ValueError):parse_bond_notice(broken)
        single=self.text('2026-03-20').replace('仅针对场内份额','仅针对基金份额')
        with self.assertRaises(ValueError):parse_bond_notice(single)

    def test_prelisting_split_not_added_to_tradable_inventory(self):
        split={'type':'split','book_date':'2017-08-04','ratio':.01}
        self.assertEqual(in_window([split],'2017-08-24'),[])
        self.assertEqual(in_window([split],'2017-08-04'),[split])
        self.assertEqual(in_window([{'type':'split','effective_date':'2020-11-09','ratio':.6945}],
                                   '2020-11-16'),[])


if __name__=='__main__':unittest.main()
