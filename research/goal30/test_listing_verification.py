"""上市日期证据的故障注入；合成数据仅在内存中，不计作投资证据。"""
import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from research.goal30.verify_listing_dates import (
    BASE, EVIDENCE, SELECTED, check_market_usage, parse_listing, save_result, validate_sources,
)


class ListingVerificationTests(unittest.TestCase):
    def test_identity_date_and_exchange_fail_closed(self):
        text = ('基金名称示例交易型开放式指数证券投资基金基金简称示例ETF'
                '基金主代码515880基金份额上市的证券交易所上海证券交易所上市日期2019年9月6日')
        self.assertEqual(parse_listing(text, 'SH515880')['issuer_list_date'], '2019-09-06')
        self.assertEqual(parse_listing(text.replace('2019年9月6日', '2019-09-06'), 'SH515880')['issuer_list_date'],
                         '2019-09-06')
        mutations = {
            'wrong_code': text.replace('515880', '510300'),
            'missing_code': text.replace('基金主代码515880', ''),
            'duplicate_code': text + '基金主代码515880',
            'wrong_exchange': text.replace('上海证券', '深圳证券'),
            'missing_date': text.replace('上市日期2019年9月6日', ''),
            'duplicate_date': text + '上市日期2019年9月6日',
            'impossible_date': text.replace('2019年9月6日', '2019年2月30日'),
            'feeder_fund': text.replace('指数证券投资基金', '指数证券投资基金联接基金'),
        }
        for label, body in mutations.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_listing(body, 'SH515880')

    def test_source_scope_failures(self):
        sources = json.loads((EVIDENCE / 'sources.json').read_text(encoding='utf-8'))['sources']
        symbols = json.loads(SELECTED.read_text(encoding='utf-8'))['symbols']
        validate_sources(sources, symbols)
        mutations = {'missing_report': sources[:-1], 'duplicate_symbol': sources[:-1] + [sources[0]]}
        for label, field, value in [
            ('outside_pool', 'symbol', 'SH513100'),
            ('wrong_document', 'file', sources[1]['file']),
            ('same_document_twice', 'sha256', sources[1]['sha256']),
            ('unapproved_host', 'url', 'https://example.com/report.pdf'),
            ('failed_response', 'status', 403),
            ('only_reminder', 'title', '2025年年度报告提示性公告'),
            ('feeder_title', 'title', '联接基金2025年年度报告'),
        ]:
            rows = copy.deepcopy(sources)
            rows[0][field] = value
            mutations[label] = rows
        for label, rows in mutations.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_sources(rows, symbols)

    def fixture(self):
        return {
            'listings': {'SH515880': {'issuer_list_date': '2019-09-06'}},
            'metadata': pd.DataFrame([{'symbol': 'SH515880', 'list_date': '2019-09-06', 'exchange': 'SH'}]),
            'bars': pd.DataFrame([{'symbol': 'SH515880', 'date': '2019-09-06'}]),
            'eligibility': pd.DataFrame([{'symbol': 'SH515880', 'date': '2019-09-05', 'eligible': False},
                                         {'symbol': 'SH515880', 'date': '2019-09-06', 'eligible': True}]),
            'cases': {'base': {
                'trades': pd.DataFrame([{'symbol': 'SH515880', 'date': '2019-09-09',
                                        'signal_date': '2019-09-06', 'side': 'buy'}]),
                'holdings': pd.DataFrame([{'symbol': 'SH515880', 'date': '2019-09-09'}]),
            }},
        }

    def test_market_usage_rejects_bad_boundaries(self):
        data = self.fixture()
        result = check_market_usage(**data)
        self.assertEqual(result['eligibility']['prelisting_rows'], 1)
        self.assertEqual(result['eligibility']['prelisting_violations'], 0)
        for label in ['metadata_date', 'duplicate_metadata', 'prelisting_price', 'duplicate_bars',
                      'prelisting_eligible', 'string_eligible', 'prelisting_signal', 'same_day_execution',
                      'prelisting_trade', 'prelisting_holding', 'outside_pool_trade']:
            data = self.fixture()
            if label == 'metadata_date':
                data['metadata'].loc[0, 'list_date'] = '2019-09-05'
            elif label == 'duplicate_metadata':
                data['metadata'] = pd.concat([data['metadata'], data['metadata']])
            elif label == 'prelisting_price':
                data['bars'].loc[0, 'date'] = '2019-09-05'
            elif label == 'duplicate_bars':
                data['bars'] = pd.concat([data['bars'], data['bars']])
            elif label == 'prelisting_eligible':
                data['eligibility'].loc[0, 'eligible'] = True
            elif label == 'string_eligible':
                data['eligibility']['eligible'] = ['False', 'True']
            elif label == 'prelisting_signal':
                data['cases']['base']['trades'].loc[0, 'signal_date'] = '2019-09-05'
            elif label == 'same_day_execution':
                data['cases']['base']['trades'].loc[0, 'signal_date'] = '2019-09-09'
            elif label == 'prelisting_trade':
                data['cases']['base']['trades'].loc[0, 'date'] = '2019-09-05'
            elif label == 'prelisting_holding':
                data['cases']['base']['holdings'].loc[0, 'date'] = '2019-09-05'
            elif label == 'outside_pool_trade':
                data['cases']['base']['trades'].loc[0, 'symbol'] = 'SH513100'
            with self.subTest(label=label), self.assertRaises(ValueError):
                check_market_usage(**data)

    def test_cannot_overwrite_input_or_write_outside_verification(self):
        for path in [SELECTED, EVIDENCE / 'sources.json', BASE / 'output/goal30/verification/listing_dates/x.pdf',
                     BASE / 'output/goal30/verification/listing_dates/../cash_replay/x.json']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                save_result(path, {'input_hashes': {}})
        path = BASE / 'output/goal30/verification/listing_dates/input.json'
        with self.assertRaises(ValueError):
            save_result(path, {'input_hashes': {path.relative_to(BASE).as_posix(): 'not-written'}})


if __name__ == '__main__':
    unittest.main()
