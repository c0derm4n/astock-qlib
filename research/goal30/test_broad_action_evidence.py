"""业务边界测试：与目视原件数值核对，并拒绝足以扭曲账户的异常输入。"""
import unittest

from research.goal30.broad_action_evidence import ROOT, split_notice, dividend_notice, compact


class PrimaryActionsTest(unittest.TestCase):
    @staticmethod
    def text(stem):
        return (ROOT / (stem + '.txt')).read_text(encoding='utf-8')

    def test_twelve_split_schedules_against_reviewed_originals(self):
        rows = [
            ('SH510500', '2022-08-26', '2022-08-26', '2022-08-26', '2022-08-29', '1.14539'),
            ('SH512010', '2021-06-28', '2021-06-24', '2021-06-25', '2021-06-28', '4'),
            ('SH512100', '2022-09-05', '2022-09-01', '2022-09-02', '2022-09-05', '0.36555'),
            ('SH512170', '2021-02-25', '2021-02-23', '2021-02-24', '2021-02-25', '3'),
            ('SH512200', '2024-08-12', '2024-08-09', '2024-08-09', '2024-08-12', '0.35806260'),
            ('SH512690', '2021-05-17', '2021-05-13', '2021-05-14', '2021-05-17', '2'),
            ('SH512800', '2025-07-07', '2025-07-04', '2025-07-04', '2025-07-07', '2'),
            ('SH515070', '2026-07-06', '2026-07-03', '2026-07-03', '2026-07-06', '2'),
            ('SH515220', '2024-04-12', '2024-04-11', '2024-04-11', '2024-04-12', '2'),
            ('SZ159901', '2021-04-12', '2021-04-09', '2021-04-09', '2021-04-12', '2'),
            ('SZ159928', '2021-06-25', '2021-06-24', '2021-06-24', '2021-06-25', '4'),
            ('SZ159995', '2026-07-07', '2026-07-06', '2026-07-06', '2026-07-07', '2'),
        ]
        for symbol, source_day, record, book, available, ratio in rows:
            with self.subTest(symbol=symbol):
                stem = f'{symbol}_split_{source_day}'
                result = split_notice(self.text(stem + '_result'), symbol, available,
                                      self.text(stem + '_plan'))['event']
                self.assertEqual((result['record_date'], result['book_date'],
                                  result['new_units_available_date'], result['ratio_decimal']),
                                 (record, book, available, ratio))

    def test_eight_dividends_against_reviewed_originals(self):
        rows = [
            ('SH512690', '2021-12-30', '2021-12-31', '2022-01-06', '0.35'),
            ('SH515220', '2021-12-24', '2021-12-27', '2021-12-30', '0.080'),
            ('SH515220', '2022-12-26', '2022-12-27', '2022-12-30', '0.040'),
            ('SH515220', '2023-12-27', '2023-12-28', '2024-01-03', '0.050'),
            ('SH515220', '2024-12-23', '2024-12-24', '2024-12-27', '0.045'),
            ('SH515220', '2025-12-23', '2025-12-24', '2025-12-29', '0.040'),
            ('SZ159901', '2025-11-14', '2025-11-17', '2025-11-19', '0.085'),
            ('SZ159920', '2018-06-22', '2018-06-22', '2018-06-26', '0.076'),
        ]
        for symbol, record, ex, pay, cash in rows:
            with self.subTest(symbol=symbol, ex=ex):
                result = dividend_notice(self.text(f'{symbol}_cash_dividend_{ex}_dividend'), symbol)['event']
                self.assertEqual((result['record_date'], result['ex_date'],
                                  result['announced_payment_date'], result['cash_per_share_decimal']),
                                 (record, ex, pay, cash))
                self.assertFalse(result['actual_receipt_confirmed'])
                self.assertIsNone(result['actual_cash_available_date'])
                self.assertNotIn('payment_date', result)  # 不可静默喂给旧执行器作为可用现金日。

    def test_same_day_entitlement_and_transfer_are_required(self):
        original = compact(self.text('SZ159920_cash_dividend_2018-06-22_dividend'))
        result = dividend_notice(original, 'SZ159920')['event']
        self.assertEqual(result['entitlement_basis'], 'holdings_before_record_day_trades')
        for phrase in ['权益登记日买入的基金份额不享有本次分红权益', '自本基金托管账户划出',
                       '投资者具体收到现金红利的时间遵循']:
            with self.subTest(phrase=phrase), self.assertRaises(ValueError):
                dividend_notice(original.replace(phrase, ''), 'SZ159920')

    def test_wrong_share_class_units_identity_and_ambiguous_dates_rejected(self):
        original = compact(self.text('SZ159901_cash_dividend_2025-11-17_dividend'))
        bad = [original.replace('(场内)', ''), original.replace('(场外)', '(场内)'),
               original.replace('元/10份', '元/100份'), original.replace('159901', '159902'),
               original.replace('易方达基金管理有限公司', '其他管理人'),
               original.replace('除息日2025年11月17日', '除息日2025年11月17日2025年11月18日')]
        for i, text in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                dividend_notice(text, 'SZ159901')

    def test_split_conflicting_ratios_rounding_and_missing_plan_rejected(self):
        original = compact(self.text('SH512100_split_2022-09-05_result'))
        for text in [original + '本基金基金份额合并比例为0.3656。',
                     original.replace('小数点后基金份额均进位为1份', '保留小数'),
                     original.replace('2022年9月2日', '2022年9月6日')]:
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                split_notice(text, 'SH512100', '2022-09-05')
        with self.assertRaises(ValueError):
            split_notice(self.text('SZ159928_split_2021-06-25_result'), 'SZ159928', '2021-06-25')

    def test_new_units_are_not_book_date_and_rounding_is_not_universal(self):
        csi = split_notice(self.text('SH510500_split_2022-08-26_result'), 'SH510500', '2022-08-29')['event']
        chip = split_notice(self.text('SZ159995_split_2026-07-07_result'), 'SZ159995', '2026-07-07')['event']
        ai = split_notice(self.text('SH515070_split_2026-07-06_result'), 'SH515070', '2026-07-06')['event']
        self.assertLess(csi['book_date'], csi['new_units_available_date'])
        self.assertEqual(csi['rounding'], 'ceiling_to_integer_manager_funded')
        self.assertEqual(chip['rounding'], 'truncate_to_integer_remainder_to_fund')
        self.assertEqual(ai['rounding'], 'registrar_exact_allocation_integer_ratio')


if __name__ == '__main__':
    unittest.main()
