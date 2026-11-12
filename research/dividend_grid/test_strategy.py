"""状态机、时序、自融资、分红和容量约束的合成测试。"""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import GridState,candidates,decide,indicators,targets
from .engine import simulate


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.case=next(c for c in candidates() if c['id']=='grid_risk_bond')
        self.row=dict(price=100.,ma_grid=100.,ma_risk=100.,slope=.01,bias=0.,vol=.15,relative40=0.)

    def test_registry(self):
        self.assertEqual(len(candidates()),20)
        self.assertEqual(len({x['id'] for x in candidates()}),20)

    def test_deep_discount_only_one_step_per_week(self):
        state=GridState();r={**self.row,'bias':-.15}
        self.assertEqual(decide(state,r,self.case)[0],.75)
        self.assertEqual(decide(state,r,self.case)[0],.95)
        self.assertEqual(decide(state,r,self.case)[0],.95)

    def test_risk_overrides_deep_discount_and_resets_grid(self):
        state=GridState(stage=2)
        w,reason=decide(state,{**self.row,'price':90.,'slope':-.01,'bias':-.15},self.case)
        self.assertEqual(w,.30);self.assertEqual(reason,'risk_cap')
        self.assertTrue(state.risk);self.assertEqual(state.stage,0)

    def test_volatility_alone_triggers_risk(self):
        state=GridState();w,_=decide(state,{**self.row,'vol':.201},self.case)
        self.assertEqual(w,.30)

    def test_recovery_needs_consecutive_confirmations(self):
        state=GridState(risk=True)
        good={**self.row,'price':102.,'bias':-.06}
        self.assertEqual(decide(state,good,self.case)[0],.30)
        self.assertEqual(decide(state,self.row,self.case)[0],.30)
        self.assertEqual(state.recovery_count,0)
        self.assertEqual(decide(state,good,self.case)[0],.30)
        w,reason=decide(state,good,self.case)
        self.assertEqual(w,.60);self.assertEqual(reason,'risk_recovery');self.assertEqual(state.stage,0)

    def test_overheat_only_removes_satellite(self):
        state=GridState(stage=2)
        self.assertEqual(decide(state,{**self.row,'bias':.10},self.case)[0],.75)
        self.assertEqual(decide(state,{**self.row,'bias':.10},self.case)[0],.60)
        self.assertEqual(decide(state,{**self.row,'bias':.10},self.case)[0],.60)

    def test_equal_risk_boundary_is_not_strict_break(self):
        state=GridState()
        w,_=decide(state,{**self.row,'price':96.,'slope':-.01,'vol':.20},self.case)
        self.assertEqual(w,.60);self.assertFalse(state.risk)

    def test_indicators_and_state_do_not_use_future(self):
        days=pd.bdate_range('2018-01-01',periods=400)
        frame=pd.DataFrame({'red':100*np.exp(np.arange(400)*.0003),'csi':100.,'bond_listed':True},index=days)
        partial=frame.iloc[:350]
        start=str(days[300].date())
        pd.testing.assert_frame_equal(targets(frame,self.case,start).loc[:partial.index[-1]],targets(partial,self.case,start))
        pd.testing.assert_frame_equal(indicators(frame,200).iloc[:350],indicators(partial,200))


class EngineTests(unittest.TestCase):
    def market(self,n=6):
        days=pd.bdate_range('2024-01-08',periods=n)
        m=pd.DataFrame({'red':10.,'bond':100.,'red_tradeable':True,'bond_tradeable':True,
            'red_open':10.,'bond_open':100.,'red_adv':1e7,'bond_adv':1e7,
            'red_amount':1e7,'bond_amount':1e7,'red_volume':1e6,'bond_volume':1e5},index=days)
        return m

    def signal(self,m,red=.60,bond=.35,risk=False):
        return pd.DataFrame({'red':[red],'bond':[bond],'risk':[risk]},index=m.index[:1])

    def test_new_position_does_not_earn_prior_rally(self):
        m=self.market();m.red=[10,12,18,18,18,18]
        d,t,_,_=simulate(m,self.signal(m,.95,0),str(m.index[0].date()),cost=0,haircut=0)
        self.assertEqual(d.nav.iloc[0],100000.)
        self.assertEqual(d.nav.iloc[1],99995.)
        self.assertEqual(str(t.date.iloc[0].date()),'2024-01-09')
        self.assertAlmostEqual(d.nav.iloc[2],147495.)

    def test_etf_round_lots_and_positive_cash(self):
        m=self.market();d,t,_,_=simulate(m,self.signal(m),str(m.index[0].date()),etf=True,actions={'red':[],'bond':[]})
        self.assertTrue(t.quantity.mod(100).eq(0).all())
        self.assertTrue(d.cash.ge(0).all())

    def test_adv_is_lagged(self):
        m=self.market();m.loc[m.index[0],'bond_adv']=1.
        d,t,_,_=simulate(m,self.signal(m,0,.95),str(m.index[0].date()),etf=True,actions={'red':[],'bond':[]})
        self.assertTrue(t.empty)

    def test_capacity_caps_actual_fills(self):
        m=self.market();m.red_adv=100000.;m.red_amount=100000.
        d,t,_,_=simulate(m,self.signal(m,.95,0),str(m.index[0].date()),etf=True,actions={'red':[],'bond':[]})
        self.assertEqual(t.quantity.iloc[0],500.)
        self.assertLessEqual(t.notional.iloc[0],5000.)

    def test_missing_quote_defers_order(self):
        m=self.market();m.loc[m.index[1],'red_tradeable']=False
        _,t,_,_=simulate(m,self.signal(m,.95,0),str(m.index[0].date()),haircut=0)
        self.assertEqual(t.date.iloc[0],m.index[2])
        self.assertEqual(t.signal_date.iloc[0],m.index[0])

    def test_split_preserves_value(self):
        m=self.market();m.loc[m.index[2]:,'red']=5.
        actions={'red':[{'effective_date':str(m.index[2].date()),'ratio':2.}], 'bond':[]}
        d,_,_,_=simulate(m,self.signal(m,.60,0),str(m.index[0].date()),etf=True,actions=actions,capacity=None)
        self.assertAlmostEqual(d.nav.iloc[1],d.nav.iloc[2])
        self.assertEqual(d.red_quantity.iloc[2],2*d.red_quantity.iloc[1])

    def test_dividend_receivable_is_not_spendable_cash(self):
        m=self.market();m.loc[m.index[3]:,'bond']=99.
        event=dict(event_id='D',record_date=str(m.index[2].date()),ex_date=str(m.index[3].date()),
                   payment_date=str(m.index[5].date()),cash_per_share=1.)
        d,t,events,_=simulate(m,self.signal(m,0,.95),str(m.index[0].date()),etf=True,
                              actions={'red':[],'bond':[event]},capacity=None)
        entitlement=d.bond_quantity.iloc[2]
        self.assertEqual(d.receivable.iloc[3],entitlement)
        self.assertEqual(d.cash.iloc[2],d.cash.iloc[3])
        self.assertEqual(d.receivable.iloc[5],0.)
        self.assertEqual(d.cash.iloc[5]-d.cash.iloc[4],entitlement)
        self.assertAlmostEqual(d.nav.iloc[2],d.nav.iloc[3])

    def test_risk_cap_ignores_small_drift(self):
        m=self.market(8)
        s=self.signal(m,.31,0)
        s.loc[m.index[3]]=[.30,0.,True]
        d,t,_,_=simulate(m,s,str(m.index[0].date()),haircut=0)
        self.assertTrue(t.side.eq('sell').any())

    def test_self_financing_identity(self):
        m=self.market();m.red=[10.,11.,12.,9.,10.,11.]
        d,_,_,_=simulate(m,self.signal(m),str(m.index[0].date()))
        expected=d.previous_nav+d.market_pnl+d.execution_pnl-d.trading_cost-d.replication_cost
        self.assertTrue(np.allclose(expected,d.nav,atol=1e-7,rtol=0))

    def test_proxy_independent_reconstruction_without_etf_flag(self):
        from .verify import reconstruct
        m=self.market().drop(columns='red_tradeable')
        d,t,_,_=simulate(m,self.signal(m),str(m.index[0].date()))
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            d.to_csv(folder/'daily.csv',index_label='date');t.to_csv(folder/'trades.csv',index=False)
            error=reconstruct(folder,m,{'red':[],'bond':[]},{'etf':False,'cost':.001,'haircut':.006})
            self.assertLess(error,1e-6)


if __name__=='__main__':unittest.main()
