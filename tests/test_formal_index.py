"""全收益指数参考的覆盖、归一化、不可交易与输入完整性检查。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import formal_index
from src.formal_rules import registry, ACTIVE_STRATEGY_IDS
from src.formal_service import stats, runtime_status, preview
from src.runs import sha256
from src.valuation.data import build_calendar


class FormalIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calendar = build_calendar('2016-09-12', '2019-01-04')
        prices = np.where(self.calendar.year == 2018, 80., 100.)
        self.frame = pd.DataFrame({'date': self.calendar, 'close': prices, 'index_code': 'H20269'})
        self.seed = self.root / 'H20269.csv'
        self.frame.to_csv(self.seed, index=False)
        self.seed.with_suffix('.metadata.json').write_text(json.dumps({'sha256': sha256(self.seed)}))

    def test_registry_contains_only_ordered_active_strategies(self):
        definitions = registry()['strategies']
        self.assertEqual(tuple(d['id'] for d in definitions), ACTIVE_STRATEGY_IDS)
        self.assertEqual([d['risk_label'] for d in definitions], ['稳健策略', '平衡策略', '积极策略'])
        self.assertNotIn('lightgbm', ACTIVE_STRATEGY_IDS)
        self.assertNotIn('chinext_pb_equal', ACTIVE_STRATEGY_IDS)

    def test_gaps_duplicates_wrong_index_and_nonfinite_values_are_rejected(self):
        wrong = self.frame.assign(index_code='H30269')
        for frame in [self.frame.drop(index=5), pd.concat([self.frame, self.frame.iloc[:1]]),
                      wrong, self.frame.assign(close=np.nan), self.frame.assign(close=0.)]:
            with self.assertRaises(ValueError):
                formal_index.validate_frame(frame, '2016-09-12', '2019-01-04')

    def test_seed_hash_and_daily_coverage_must_pass_before_build(self):
        out = self.root / 'run'
        out.mkdir()
        with patch.object(formal_index, 'SEED', self.seed):
            result = formal_index.prepare_data(out, '2016-09-12', pd.Timestamp('2019-01-04'))
            self.assertEqual(len(result), len(self.calendar))
            self.seed.write_text('damaged', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '哈希'):
                formal_index.prepare_data(self.root, '2016-09-12', pd.Timestamp('2019-01-04'))

    def test_reference_keeps_2018_loss_without_orders_or_false_cash(self):
        frame = formal_index.validate_frame(self.frame, '2016-09-12', '2019-01-04')
        # 旧指数引擎仅作归档回归；活动登记已切换为A。
        definition = {'id':'dividend_lowvol_total_return','kind':'index_benchmark',
            'name':'归档红利全收益参考','color':'#a84c54','description':'归档指数持有参考',
            'account_basis':'100%全收益指数持有参考'}
        payload = formal_index.build_strategy(self.root, definition, registry(), frame, 'test', stats)
        self.assertEqual(payload['series']['capital'][0], 1e6)
        self.assertAlmostEqual(payload['metrics']['max_drawdown'], .2)
        self.assertAlmostEqual(payload['metrics']['worst_complete_year']['return'], -.2)
        self.assertEqual(payload['metrics']['worst_complete_year']['year'], 2018)
        self.assertEqual(payload['holdings'], [])
        self.assertEqual(payload['advice']['rows'], [])
        self.assertEqual(payload['validation']['trade_count'], 0)
        self.assertIsNone(payload['validation']['weekly_t_plus_1'])
        self.assertFalse(payload['capabilities']['account_preview'])
        self.assertEqual(payload['cash'], 0.)
        self.assertFalse(payload['investment_admission'])
        with self.assertRaisesRegex(ValueError, '指数'):
            preview({}, definition, 100000, [])
        status = runtime_status(payload, pd.Timestamp('2019-01-04 20:00', tz='Asia/Shanghai'))
        self.assertEqual(status['advice_status'], 'reference')
        self.assertIsNone(status['execution_date'])
        self.assertFalse(status['can_execute'])
        status = runtime_status(payload, pd.Timestamp('2019-01-07 20:00', tz='Asia/Shanghai'))
        self.assertTrue(status['stale'])


if __name__ == '__main__':
    unittest.main()
