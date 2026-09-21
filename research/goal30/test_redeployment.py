"""容量、组别、风险、因果与退出边界的重新分配测试。"""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.group_diversification import GROUP_OF, VARIANTS, covariance, execution_policy, decision as old_decision
from research.goal30.redeployment import capped_proportional, position_redistribution, decision
from research.goal30.reconcile_redeployment import independent_positions, independent_targets
from src.portfolio import risk_group


class RedeploymentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(814)
        self.prices=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001,.012,(330,30)),axis=0)),
            columns=sorted(GROUP_OF),index=pd.bdate_range('2020-01-02',periods=330))
        self.policy=execution_policy(json.loads(SELECTED.read_text(encoding='utf-8'))['policy'])

    def test_manual_capacity_redistribution_and_insufficient_capacity(self):
        result=capped_proportional({'A':.8,'B':.1,'C':.1},{'A':.35,'B':.35,'C':.35},.95)
        np.testing.assert_allclose([result[k] for k in 'ABC'],[.35,.3,.3],atol=1e-12)
        result=capped_proportional({'A':.9,'B':.1},{'A':.35,'B':.35},.95)
        np.testing.assert_allclose([result['A'],result['B']],[.35,.35],atol=1e-12)
        self.assertEqual(capped_proportional({}, {}, .95),{})
        for weights,caps,budget in [({'A':-1},{'A':.35},.5),({'A':1},{'B':.35},.5),
                                    ({'A':1},{'A':-.1},.5),({'A':1},{'A':.35},float('nan'))]:
            with self.assertRaises(ValueError): capped_proportional(weights,caps,budget)

    def test_group_capacity_and_original_budget_preserved(self):
        raw={'SH515880':.4,'SH515070':.3,'SH512720':.1,'SH518880':.15}
        result=position_redistribution(raw,self.policy)
        self.assertAlmostEqual(sum(result.values()),.95)
        self.assertAlmostEqual(result['SH518880'],.25)
        self.assertAlmostEqual(sum(result[s] for s in raw if risk_group(s)=='growth'),.7)
        reduced=position_redistribution({'SH518880':.2,'SH512800':.1},self.policy)
        self.assertAlmostEqual(sum(reduced.values()),.3)
        with self.assertRaises(ValueError): position_redistribution({'SPY':.9},self.policy)

    def test_independent_bisection_and_input_order_on_random_weights(self):
        rng=np.random.default_rng(338)
        for _ in range(100):
            symbols=list(rng.choice(sorted(GROUP_OF),size=int(rng.integers(1,16)),replace=False))
            raw=dict(zip(symbols,rng.lognormal(-1,2,len(symbols))))
            actual=position_redistribution(raw,self.policy);independent=independent_positions(raw,self.policy)
            self.assertEqual(set(actual),set(independent))
            np.testing.assert_allclose([actual[s] for s in actual],[independent[s] for s in actual],atol=1e-11)
            reversed_input=dict(reversed(list(raw.items())))
            self.assertEqual(actual,position_redistribution(reversed_input,self.policy))

    def test_unconstrained_case_equals_original_proportions(self):
        raw={'SH518880':.2,'SH512800':.15,'SH510300':.25}
        result=position_redistribution(raw,self.policy)
        np.testing.assert_allclose([result[s] for s in raw],list(raw.values()),atol=1e-12)

    def test_high_volatility_and_drift_limit_rechecked(self):
        prices=self.prices*1.
        rng=np.random.default_rng(15)
        prices.iloc[:]=100*np.exp(np.cumsum(rng.normal(.001,.06,prices.shape),axis=0))
        scores={'SH515880':3.,'SH512800':2.,'SH518880':1.,'SH511260':.5}
        state,target,trace=decision({},scores,prices,set(scores),[],{},self.policy,VARIANTS[0])
        expected=independent_targets(state['raw_weights'],prices,{},self.policy)
        np.testing.assert_allclose([target[s] for s in target],[expected[0][s] for s in target],atol=1e-10)
        for name in ('first_risk_limited_weights','before_turnover_weights'):
            weights=trace[name];symbols=sorted(weights);v=np.array([weights[s] for s in symbols])
            self.assertLessEqual(float(np.sqrt(v@covariance(prices,symbols)@v)),.15+1e-10)
        self.assertLessEqual(sum(target.values()),self.policy['max_daily_turnover']+1e-10)

    def test_monthly_exit_restore_and_future_prefix(self):
        scores={'SH515880':3.,'SH512800':2.,'SH518880':1.}
        prefix=self.prices.iloc[:280]
        state,target,trace=decision({},scores,prefix,set(scores),[],{},self.policy,VARIANTS[0])
        old=old_decision({},scores,prefix,set(scores),[],{},self.policy,VARIANTS[0])[0]
        self.assertEqual(state,old)
        future=self.prices.copy();future.iloc[280:]*=.001
        self.assertEqual((state,target,trace),decision({},scores,future.iloc[:280],set(scores),[],{},self.policy,VARIANTS[0]))
        state2,target2,_=decision(json.loads(json.dumps(state)),{s:v for s,v in scores.items() if s!='SH515880'},
            self.prices.iloc[:281],set(scores),[],target,self.policy,VARIANTS[0])
        self.assertNotIn('SH515880',state2['raw_weights']);self.assertNotIn('SH515880',target2)
        snapshot=copy.deepcopy(state2)
        state3,_,_=decision(state2,scores,self.prices.iloc[:282],set(scores),[],target2,self.policy,VARIANTS[0])
        self.assertNotIn('SH515880',state3['raw_weights']);self.assertEqual(state2,snapshot)


if __name__=='__main__': unittest.main(verbosity=2)
