import json
import tempfile
import unittest
from pathlib import Path

from research.valuation_bands import etfrun_data, run


class EqualWeightDataTests(unittest.TestCase):
    @staticmethod
    def html(rows):
        payload = '11:' + json.dumps(['$', 'section', None, {'indexDailyItems': rows}])
        return '<script>self.__next_f.push(' + json.dumps([1, payload]) + ')</script>'

    def test_extracts_raw_daily_values_without_using_provider_quantiles(self):
        frame = etfrun_data.extract_daily(self.html([
            {'date': '2011-01-05', 'close': 100, 'equalWeightedPbTtm': 2.5, 'historyPbPercentile': .99},
            {'date': '2011-01-04', 'close': 99, 'equalWeightedPbTtm': 2.4, 'historyPbPercentile': .98}]))
        self.assertEqual(frame.iloc[0]['equalWeightedPbTtm'], 2.4)
        self.assertEqual(etfrun_data.FIELDS['399006'][1], 'PB_ETFRUN_EQUAL')

    def test_duplicate_dates_rejected(self):
        row = {'date': '2011-01-04', 'close': 99, 'equalWeightedPeTtm': 20}
        with self.assertRaisesRegex(ValueError, '日期重复'):
            etfrun_data.extract_daily(self.html([row, row]))

    def test_private_or_missing_history_not_invented(self):
        with self.assertRaisesRegex(ValueError, '未找到'):
            etfrun_data.extract_daily('<html>Pro用户登录查看</html>')

    def test_alternative_protocol_hash_is_used_without_changing_original(self):
        original = run.code_hashes()['protocol.json']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'alternate.json'
            doc = run.protocol()
            doc['protocol_id'] = 'alternate'
            path.write_text(json.dumps(doc), encoding='utf-8')
            self.assertEqual(run.protocol(path)['protocol_id'], 'alternate')
            self.assertNotEqual(run.code_hashes(path)['protocol.json'], original)
        self.assertEqual(run.code_hashes()['protocol.json'], original)


if __name__ == '__main__':
    unittest.main()
