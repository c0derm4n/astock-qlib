"""逐日状态恢复、权益归属及持久化失败边界。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.goal30.audit import SELECTED
from research.goal30.daily_cash_ledger import EventCalendar, freeze_next, new_state, seal, settle_day
from research.goal30.daily_simulation import MarketDays, read_store, run_model_day
from research.sp500.explore import allocator


class DailyLedgerTests(unittest.TestCase):
    def setUp(self):
        self.selected = json.loads(SELECTED.read_text(encoding='utf-8'))
        self.selected['policy']['initial_capital'] = 100000.
        self.policy = self.selected['policy']
        self.days = pd.bdate_range('2025-01-02', periods=14)

    def market(self, end=None, dividend=False):
        days = self.days if end is None else self.days[:end+1]
        rows = []
        for symbol in ('SH510300', 'SH515880', 'SH518880'):
            for i, day in enumerate(days):
                price = 9.8 if dividend and i >= 3 and symbol == 'SH515880' else 10.
                rows.append({'date': day, 'symbol': symbol, 'open': price, 'close': price,
                             'adj_close': 10., 'volume': 1e7})
        bars = pd.DataFrame(rows)
        market = SimpleNamespace(calendar=days, bars=bars, fingerprint='synthetic-'+str(len(days)),
            eligibility=bars[['date','symbol']].assign(eligible=True, avg_amount=1e8))
        market.wide = lambda k: market.bars.pivot(index='date', columns='symbol', values=k)
        return market

    def dividend(self):
        return {'event_id':'test-dividend','symbol':'SH515880','type':'cash_dividend','cash_per_share':.2,
                'record_date':str(self.days[2].date()),'ex_date':str(self.days[3].date()),
                'payment_date':str(self.days[6].date()),'source':'synthetic test'}

    def drive(self, targets, actions=(), *, dividend=False, restore=True):
        frames = MarketDays(self.market(dividend=dividend))
        catalog = EventCalendar(actions, self.days)
        state = new_state(100000., self.days[0])
        rows=[]
        for i, day in enumerate(self.days[1:], 1):
            state, output = settle_day(state, frames.context(day), self.policy, catalog)
            rows.append((copy.deepcopy(state), output))
            if i+1 < len(self.days):
                state = freeze_next(state, targets(i), frames.close.loc[day].to_dict(),
                                    frames.adv.loc[day].to_dict(), self.policy, self.days[i+1])
            if restore:
                state = json.loads(json.dumps(state, allow_nan=False))
        return rows

    def test_dividend_belongs_to_record_holder_after_sale_and_is_paid_once(self):
        rows=self.drive(lambda i: {'SH515880':.85} if i < 3 else {}, [self.dividend()], dividend=True)
        by_day={r['report']['date']:(s,r) for s,r in rows}
        record=by_day[str(self.days[2].date())][0]
        amount=record['quantities']['SH515880']*.2
        self.assertGreater(amount, 0)
        ex=by_day[str(self.days[3].date())]
        self.assertEqual(ex[1]['report']['dividend_receivable'], amount)
        self.assertEqual(ex[1]['trades'], [])  # 除息日撤销旧订单
        paid=by_day[str(self.days[6].date())]
        self.assertNotIn('SH515880', paid[0]['quantities'])
        self.assertEqual(paid[1]['report']['dividend_receivable'], 0)
        payments=[e for _,r in rows for e in r['actions'] if e['type']=='pay_dividend']
        self.assertEqual(len(payments),1)
        self.assertEqual(payments[0]['amount'],amount)
        before=by_day[str(self.days[5].date())][0]['cash']
        self.assertAlmostEqual(paid[0]['cash']-before,amount)

    def test_buyer_after_record_gets_no_old_dividend(self):
        rows=self.drive(lambda i: {'SH515880':.85} if i >= 3 else {}, [self.dividend()], dividend=True)
        payments=[e for _,r in rows for e in r['actions'] if e['type']=='pay_dividend']
        self.assertEqual(payments[0]['amount'],0)
        self.assertTrue(any(s['quantities'] for s,_ in rows))

    def test_recovering_json_each_day_does_not_change_execution_or_reenter_exit(self):
        target=lambda i: {'SH515880':.85} if i < 6 else {}
        recovered=self.drive(target,restore=True)
        continuous=self.drive(target,restore=False)
        self.assertEqual(recovered,continuous)
        self.assertEqual(recovered[-1][0]['quantities'],{})
        self.assertTrue(all(r['report']['budget_turnover'] <= .3 for _,r in recovered))
        sales=[r['report']['date'] for _,r in recovered if any(t['side']=='sell' for t in r['trades'])]
        self.assertGreaterEqual(len(sales),3)

    def test_receivable_is_not_buying_power(self):
        state=new_state(100.,self.days[0])
        state.update({'receivable':{'x':100000.},'previous_nav':100100.})
        state=seal(state)
        frozen=freeze_next(state,{'SH515880':.85},{'SH515880':10.},{'SH515880':1e8},self.policy,self.days[1])
        self.assertEqual(frozen['pending_orders'],[])

    def test_fixed_target_rebuild_matches_persistent_allocator_between_rebalance_days(self):
        frames=MarketDays(self.market())
        persistent=allocator(self.selected['candidate'],self.days,self.days[1])
        state=new_state(100000.,self.days[0])
        catalog=EventCalendar([],self.days)
        for i,day in enumerate(self.days[1:],1):
            state,_=settle_day(state,frames.context(day),self.policy,catalog)
            values={s:q*frames.close.at[day,s] for s,q in state['quantities'].items()}
            nav=state['cash']+sum(values.values())
            expected=persistent(pd.Series(dtype=float),list(frames.eligible.columns),
                {s:v/nav for s,v in values.items()}, {},frames.returns.loc[:day],self.policy,exiting=state['exiting'])
            actual=frames.fixed_targets(self.selected,state,self.days[1])
            self.assertEqual(actual,expected)
            self.assertTrue(actual)
            if i+1<len(self.days):
                state=freeze_next(state,actual,frames.close.loc[day].to_dict(),frames.adv.loc[day].to_dict(),
                                  self.policy,self.days[i+1])
            state=json.loads(json.dumps(state))

    def test_mutation_duplicate_day_and_late_event_rejected_atomically(self):
        frames=MarketDays(self.market())
        state=new_state(100000.,self.days[0])
        empty=EventCalendar([],self.days)
        closed,_=settle_day(state,frames.context(self.days[1]),self.policy,empty)
        snapshot=copy.deepcopy(closed)
        bad=copy.deepcopy(closed);bad['cash']+=1
        with self.assertRaisesRegex(ValueError,'指纹'):
            settle_day(bad,frames.context(self.days[2]),self.policy,empty)
        with self.assertRaisesRegex(ValueError,'日期'):
            settle_day(closed,frames.context(self.days[1]),self.policy,empty)
        late=self.dividend();late['record_date']=str(self.days[1].date())
        with self.assertRaisesRegex(ValueError,'已结算日期'):
            settle_day(closed,frames.context(self.days[2]),self.policy,EventCalendar([late],self.days))
        self.assertEqual(closed,snapshot)
        catalog=EventCalendar([self.dividend()],self.days)
        first,_=settle_day(state,frames.context(self.days[1]),self.policy,catalog)
        changed=self.dividend();changed['cash_per_share']=.3
        with self.assertRaisesRegex(ValueError,'删除或修改'):
            settle_day(first,frames.context(self.days[2]),self.policy,EventCalendar([changed],self.days))

    def test_split_and_suspension_survive_restart(self):
        market=self.market()
        day=self.days[5]
        mask=market.bars.symbol.eq('SH515880') & market.bars.date.ge(day)
        market.bars.loc[mask,['open','close']]/=1.015
        market.bars.loc[market.bars.date.eq(self.days[7]),'volume']=0.
        frames=MarketDays(market)
        catalog=EventCalendar([{'event_id':'split','symbol':'SH515880','type':'split','ratio':1.015,
                                'effective_date':str(day.date())}],self.days)
        state=new_state(100000.,self.days[0]);trades=[]
        for i,date in enumerate(self.days[1:],1):
            state,output=settle_day(state,frames.context(date),self.policy,catalog)
            trades+=output['trades']
            if i==7:
                self.assertEqual(output['report']['budget_turnover'],0)
            if i+1<len(self.days):
                state=freeze_next(state,{'SH515880':.85} if i<6 else {},frames.close.loc[date].to_dict(),
                                  frames.adv.loc[date].to_dict(),self.policy,self.days[i+1])
            state=json.loads(json.dumps(state))
        self.assertEqual(state['quantities'],{})
        self.assertTrue(any(t['side']=='sell' and t['quantity']%100 != 0 for t in trades))

    def feed(self,i):
        day=str(self.days[i].date())
        return {'as_of':day,'covered_through':day,'source_note':'synthetic test only','events':[]}

    def test_daily_store_idempotency_and_rejects_stale_gap_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json'
            call=lambda i,**kw: run_model_day(path,self.selected,self.market(i),self.feed(i),
                now=str(self.days[i].date())+'T19:02:00+08:00',**kw)
            first=call(1,initial_cash=100000.)
            before=path.read_bytes()
            self.assertTrue(call(1)['already_recorded'])
            self.assertEqual(path.read_bytes(),before)
            changed=self.feed(1);changed['source_note']='changed'
            with self.assertRaisesRegex(ValueError,'输入变化'):
                run_model_day(path,self.selected,self.market(1),changed,now=str(self.days[1].date())+'T19:02:00+08:00')
            with self.assertRaisesRegex(ValueError,'漏跑'):
                call(3)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'行情未更新'):
                run_model_day(path,self.selected,self.market(1),self.feed(2),now=str(self.days[2].date())+'T19:02:00+08:00')
            second=call(2)
            self.assertEqual(second['qualified_forward_days'],0)
            self.assertFalse(second['executable'])
            self.assertEqual(second['orders'],[])
            self.assertEqual(len(read_store(path)['sessions']),2)
            damaged=json.loads(path.read_text(encoding='utf-8'));damaged['sessions'][0]['state']['cash']+=1
            path.write_text(json.dumps(damaged),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'指纹'):
                read_store(path)


if __name__ == '__main__':
    unittest.main()
