"""分组机制、风险预算、因果时点及月内退出的经济边界。"""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from research.goal30.group_diversification import (GROUP_OF, VARIANTS, covariance, decision,
    execution_policy, feature_matrix, limit_weights, prefix_scores, risk_parity, select_symbols)
from research.goal30.audit import SELECTED


class GroupDiversificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(45)
        self.prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001, .009, (330, 30)), axis=0)),
            columns=sorted(GROUP_OF), index=pd.bdate_range('2020-01-02', periods=330))
        self.policy = execution_policy(json.loads(SELECTED.read_text(encoding='utf-8'))['policy'])

    def test_full_features_equal_independent_prefix_and_future_invariant(self):
        features = feature_matrix(self.prices)
        for i in (250, 252, 280, 300):
            expected = prefix_scores(self.prices.iloc[:i+1])
            actual = features.iloc[i].dropna().to_dict()
            self.assertEqual(set(actual), set(expected))
            np.testing.assert_allclose([actual[s] for s in expected], list(expected.values()), rtol=1e-12, atol=1e-12)
        altered = self.prices.copy(); altered.iloc[290:] *= .05
        pd.testing.assert_frame_equal(feature_matrix(altered).iloc[:290], features.iloc[:290])
        scaled = self.prices * np.linspace(.1, 100, 30)
        np.testing.assert_allclose(feature_matrix(scaled), features, atol=1e-11, rtol=1e-11, equal_nan=True)

    def test_missing_prices_not_filled_and_mixed_horizons_rejected(self):
        bad = self.prices.copy()
        bad.iloc[260, 0] = np.nan
        bad.iloc[300, 1] = 0
        features = feature_matrix(bad)
        self.assertTrue(features.iloc[260:, 0].isna().all())
        self.assertTrue(features.iloc[300:, 1].isna().all())
        with self.assertRaisesRegex(ValueError, '连续真实'):
            covariance(bad, [bad.columns[1]])
        decline = pd.DataFrame(np.repeat(np.linspace(200, 100, 330)[:, None], 30, axis=1),
                               index=self.prices.index, columns=self.prices.columns)
        self.assertTrue(feature_matrix(decline).isna().all().all())

    def test_group_and_boundary_ties_do_not_pick_code(self):
        scores = {'SH515880': 3., 'SH515070': 2., 'SH512800': 1.5, 'SH518880': 1.}
        grouped = select_symbols(scores, set(scores), VARIANTS[0])
        self.assertIn('SH515880', grouped); self.assertNotIn('SH515070', grouped)
        ties = {**scores, 'SH515070': 3.}
        self.assertNotIn('SH515880', select_symbols(ties, set(ties), VARIANTS[0]))
        self.assertEqual(select_symbols({'SH512800': 1., 'SH518880': 1.}, {'SH512800', 'SH518880'},
                                       {**VARIANTS[0], 'slots': 1}), [])
        self.assertEqual(select_symbols(scores, {'SH518880'}, VARIANTS[0]), ['SH518880'])

    def test_risk_contributions_and_low_volatility_cap(self):
        matrix = np.diag([.01, .04, .09])
        np.testing.assert_allclose(risk_parity(matrix), np.array([1., .5, 1/3])/(1+.5+1/3), atol=1e-9)
        c = np.array([[.04, -.012, .005], [-.012, .09, .004], [.005, .004, .01]])
        w = risk_parity(c)
        np.testing.assert_allclose(w * (c@w) / (w@c@w), np.ones(3)/3, atol=1e-8)
        with self.assertRaisesRegex(ValueError, '输入无效'):
            risk_parity(np.array([[1., 2.], [2., 1.]]))
        limited = limit_weights({'SH515880': .6, 'SH515070': .3, 'SH518880': .4}, self.prices, self.policy)
        self.assertTrue(all(w <= .35+1e-12 for w in limited.values()))
        self.assertLessEqual(limited['SH515880']+limited['SH515070'], .7+1e-12)
        self.assertLessEqual(sum(limited.values()), .95+1e-12)

    def test_monthly_review_daily_exit_no_reentry_and_json_restore(self):
        prices = self.prices.iloc[:280]
        scores = {'SH515880': 3., 'SH512800': 2., 'SH518880': 1.}
        state, targets, trace = decision({}, scores, prices, set(scores), [], {}, self.policy, VARIANTS[0])
        self.assertTrue(trace['monthly_review'])
        self.assertEqual(len(state['raw_weights']), 3)
        self.assertAlmostEqual(sum(state['raw_weights'].values()), .95*3/4)
        current = targets.copy()
        day2 = self.prices.iloc[:281]
        state2, target2, info = decision(json.loads(json.dumps(state)), {k:v for k,v in scores.items() if k != 'SH515880'},
            day2, set(scores), [], current, self.policy, VARIANTS[0])
        self.assertFalse(info['monthly_review'])
        self.assertNotIn('SH515880', state2['raw_weights']); self.assertNotIn('SH515880', target2)
        state3, _, _ = decision(state2, scores, self.prices.iloc[:282], set(scores), [], target2, self.policy, VARIANTS[0])
        self.assertNotIn('SH515880', state3['raw_weights'])
        snapshot = copy.deepcopy(state2)
        decision(state2, scores, self.prices.iloc[:282], [], [], target2, self.policy, VARIANTS[0])
        self.assertEqual(state2, snapshot)


if __name__ == '__main__':
    unittest.main()
