"""正式A的研究等价、信息时点、独立账本与接口语义。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from src import formal_dividend as module
from src.formal_rules import registry, ACTIVE_STRATEGY_IDS
from src.formal_service import stats, next_session, runtime_status, preview
from src.runs import sha256
from src.valuation.data import build_calendar


class FormalDividendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.definition = registry()['strategies'][2]
        self.spec = registry()
        self.calendar = build_calendar('2014-01-02','2017-09-29')
        self.panel = self.make_panel(.045)

    def make_panel(self, rate):
        n = len(self.calendar)
        return pd.DataFrame({
            'red':100*np.cumprod(np.r_[1.,np.full(n-1,1+rate/252)]),
            'price':100.,'bond':np.where(self.calendar>='2017-08-24',100.,1.),
            'bond_listed':self.calendar>=pd.Timestamp('2017-08-24')},index=self.calendar)

    def test_active_replacement_preserves_other_strategies_and_capital(self):
        self.assertEqual(ACTIVE_STRATEGY_IDS,('csi300_pe_equal','trend200_vol15','dividend_band_a'))
        self.assertEqual(self.spec['initial_capital'],1000000)
        self.assertEqual(self.definition['kind'],'dividend_band')
        self.assertIn('dividend_lowvol_total_return',self.spec['retired_strategy_ids'])
        self.assertEqual(self.definition['policy']['band_table'],module.BANDS)

    def test_all_seven_yield_zones_and_prelisting_cash(self):
        for rate, expected in [(.06,.95),(.049,.85),(.045,.70),(.041,.60),(.039,.50),(.037,.35),(.03,.20)]:
            p=self.make_panel(rate)
            s=module.signal_frame(p,self.definition).loc['2016-09-12':]
            self.assertTrue(s.red_target.eq(expected).all(),rate)
            self.assertTrue(s.loc[:'2017-08-23','bond_target'].eq(0).all())
            np.testing.assert_allclose(s.loc['2017-08-24':,'bond_target'],.95-expected)
            np.testing.assert_allclose(s[['red_target','bond_target','cash_target']].sum(axis=1),1.)

    def test_signal_prefix_is_unchanged_when_future_prices_change(self):
        date='2017-03-31'
        full=module.signal_frame(self.panel,self.definition)
        truncated=module.signal_frame(self.panel.loc[:date],self.definition)
        pd.testing.assert_frame_equal(full.loc[:date],truncated)
        changed=self.panel.copy()
        changed.loc[changed.index>date,['red','price']]*=2
        pd.testing.assert_frame_equal(module.signal_frame(changed,self.definition).loc[:date],truncated)

    def test_missing_duplicate_and_nonpositive_prices_are_rejected(self):
        for frame in [self.panel.drop(self.panel.index[100]),pd.concat([self.panel,self.panel.iloc[:1]]),
                      self.panel.assign(price=0),self.panel.assign(red=np.nan)]:
            with self.assertRaises(ValueError):
                module.validate_panel(frame,self.calendar[-1])

    def test_seed_hash_is_verified_before_consumption(self):
        seed=self.root/'seed.csv'
        self.panel.to_csv(seed,index_label='date')
        seed.with_suffix('.metadata.json').write_text(json.dumps({'sha256':sha256(seed)}))
        market=SimpleNamespace(calendar=self.calendar,wide=lambda field:pd.DataFrame(
            {'SH511260':100. if field=='close' else 1.},index=self.calendar))
        run=self.root/'run';run.mkdir()
        with patch.object(module,'SEED',seed):
            actual=module.prepare_data(run,market)
            self.assertEqual(actual.index.tolist(),self.calendar.tolist())
            seed.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'哈希'):
                module.prepare_data(self.root,market)

    def test_official_wrong_index_is_rejected(self):
        response=Mock()
        response.json.return_value={'code':200,'data':[{'indexCode':'OTHER','indexNameCnAll':'wrong','tradeDate':'20170929','close':100.}]}
        session=Mock();session.get.return_value=response
        market=SimpleNamespace(calendar=self.calendar)
        with patch.object(module.requests,'Session',return_value=session):
            with self.assertRaisesRegex(ValueError,'身份'):
                module._extend_panel(self.panel,market,self.root,True)

    def test_no_same_day_gain_and_independent_fee_check(self):
        dates=pd.bdate_range('2026-11-09',periods=3)
        prices=pd.DataFrame({'red':[100.,200.,220.],'bond':100.},index=dates)
        target=pd.DataFrame({'red':.70,'bond':0.},index=dates)
        daily,trades=module.replay(prices,target,str(dates[0].date()),haircut=0)
        self.assertEqual(float(daily.nav.iloc[0]),100000.)
        self.assertAlmostEqual(float(daily.nav.iloc[1]),99930.)
        self.assertAlmostEqual(float(daily.nav.iloc[2]),106930.)
        policy={**self.definition['policy'],'replication_fee':0}
        self.assertLess(module.verify_account(prices,daily,trades,100000.,policy),1e-6)
        trades.loc[0,'cost']+=10
        with self.assertRaisesRegex(ValueError,'费用'):
            module.verify_account(prices,daily,trades,100000.,policy)

    def test_payload_has_dynamic_allocations_and_no_false_etf_orders(self):
        payload=module.build_strategy(self.root,self.definition,self.spec,self.panel,'test',stats,next_session)
        self.assertEqual(payload['series']['capital'][0],1000000.)
        self.assertFalse(payload['reference_only'])
        self.assertFalse(payload['capabilities']['account_preview'])
        self.assertFalse(payload['capabilities']['executable_orders'])
        self.assertEqual(payload['advice']['execution_basis'],'next_session_close')
        self.assertTrue(payload['weekly_holdings'])
        self.assertTrue(all(h['quantity'] is None and h['equivalent'] for h in payload['holdings']))
        self.assertTrue(payload['validation']['weekly_t_plus_1'])
        self.assertEqual(payload['validation']['ledger']['status'],'passed')
        with self.assertRaisesRegex(ValueError,'等值'):
            preview({},self.definition,100000,[])

    def test_runtime_uses_close_window_for_a_and_keeps_open_for_trend(self):
        payload={'kind':'dividend_band','data_end':'2026-11-09','advice':{'signal_date':'2026-11-09'}}
        at=pd.Timestamp('2026-11-10 10:00',tz='Asia/Shanghai')
        self.assertEqual(runtime_status(payload,at)['advice_status'],'pending')
        self.assertIn('收盘',runtime_status(payload,at)['advice_label'])
        self.assertEqual(runtime_status(payload,at+pd.Timedelta(hours=6))['advice_status'],'elapsed')
        payload['kind']='fixed'
        self.assertEqual(runtime_status(payload,at)['advice_status'],'elapsed')

    def test_frozen_100k_account_is_reproduced_without_research_imports(self):
        source=module.ROOT/'output/dividend_band/runs/20261111-174236-69e76e21/cases/abs_band/daily.csv'
        if not source.exists():
            self.skipTest('冻结研究数据未随代码分发')
        p=pd.read_csv(module.SEED,index_col='date',parse_dates=True).loc[:'2026-11-04']
        s=module.signal_frame(p,self.definition)
        t=s[['red_target','bond_target']].set_axis(['red','bond'],axis=1)
        daily,trades=module.replay(p[['red','bond']],t,'2016-09-12',no_fill_dates=['2018-08-17'])
        expected=pd.read_csv(source,index_col='date',parse_dates=True)
        for name in ['nav','cash','red_value','bond_value','trading_cost','replication_cost']:
            np.testing.assert_allclose(daily[name],expected[name],rtol=0,atol=1e-6)
        self.assertEqual(len(trades),189)
        self.assertLess(module.verify_account(p,daily,trades,100000.,self.definition['policy']),1e-6)


if __name__=='__main__':
    unittest.main()
