"""同一现金/份额引擎：指数等值代理或ETF整手、分红应收与到账。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

ASSETS=('red','bond')


def simulate(market,signals,start,actions=None,*,etf=False,cost=.001,delay=1,
             haircut=.006,capital=100000.,capacity=.05,band=.03,open_fill=False,
             carry_gap=False):
    market=market.copy()
    if etf:
        for asset in ASSETS:
            market[asset+'_adv']=market[asset+'_adv'].shift(1)
    path=market.loc[start:].copy()
    if carry_gap and pd.Timestamp('2018-08-17') in path.index:
        pos=path.index.get_loc('2018-08-17')
        path.loc['2018-08-17','bond']=path.bond.iloc[pos-1]
    if not np.isfinite(path[list(ASSETS)]).all().all() or not path[list(ASSETS)].gt(0).all().all():
        raise ValueError('持仓估值缺失或非正')
    cash=capital;q=np.zeros(2);receivables={};entitlements={}
    queue={};daily=[];trades=[];event_log=[];order_log=[]
    records={};exdates={};payments={};splits={}
    if etf:
        for event in actions['bond']:
            records.setdefault(pd.Timestamp(event['record_date']),[]).append(event)
            exdates.setdefault(pd.Timestamp(event['ex_date']),[]).append(event)
            payments.setdefault(pd.Timestamp(event['payment_date']),[]).append(event)
        for event in actions['red']:
            splits[pd.Timestamp(event['effective_date'])]=float(event['ratio'])
    previous_nav=capital
    for i,day in enumerate(path.index):
        row=path.loc[day];prices=row[list(ASSETS)].to_numpy(dtype=float)
        dividend_accrued=dividend_paid=0.
        if day in splits:
            before=q[0];q[0]*=splits[day]
            event_log.append(dict(date=day,type='split',quantity_before=before,quantity_after=q[0],amount=0.))
        for event in exdates.get(day,[]):
            amount=entitlements.get(event['event_id'],0.)
            receivables[event['event_id']]=amount;dividend_accrued+=amount
            event_log.append(dict(date=day,type='dividend_accrue',event_id=event['event_id'],amount=amount))
        for event in payments.get(day,[]):
            amount=receivables.pop(event['event_id'],0.)
            cash+=amount;dividend_paid+=amount
            event_log.append(dict(date=day,type='dividend_pay',event_id=event['event_id'],amount=amount))
        before_trades=float(cash+q@prices+sum(receivables.values()))
        days=(day-path.index[i-1]).days if i else 0
        management=0.
        if not etf and days:
            reduction=q[0]*(1-math.exp(-haircut*days/365.2425))
            management=reduction*prices[0];q[0]-=reduction
        fee_today=0.;execution_pnl=0.;clipped=0
        if i in queue:
            signal_day,goal,risk=queue.pop(i)
            can_trade=np.array([bool(row.get(a+'_tradeable',True)) for a in ASSETS])
            if not can_trade.all() and any(q[j]>0 or goal[j]>0 for j in range(2) if not can_trade[j]):
                queue[i+1]=(signal_day,goal,risk)
                order_log.append(dict(date=day,signal_date=signal_day,status='quote_missing_deferred',asset='both',requested=0.,filled=0.))
            else:
                nav=float(cash+q@prices+sum(receivables.values()))
                current=q*prices/nav
                goal=goal.copy()
                keep=(current>0)&(goal>0)&(np.abs(current-goal)<band)
                if risk and current[0]>goal[0]:keep[0]=False
                goal[keep]=current[keep]
                if goal.sum()>.95:goal*=.95/goal.sum()
                fills=np.array([float(row[a+'_open']) if etf and open_fill else prices[j]
                                for j,a in enumerate(ASSETS)])
                desired=goal*nav/fills
                for side in ('sell','buy'):
                    for j,asset in enumerate(ASSETS):
                        difference=(q[j]-desired[j]) if side=='sell' else (desired[j]-q[j])
                        if difference<=1e-9 or not can_trade[j]:continue
                        requested=difference
                        lot=100 if etf else 0
                        quantity=(math.floor(difference/100)*100 if lot else difference)
                        cap=math.inf
                        if etf and capacity is not None:
                            adv=float(row[asset+'_adv']);amount=float(row[asset+'_amount'])
                            cap=max(0.,min(adv,amount)*capacity) if np.isfinite(adv) and np.isfinite(amount) else 0.
                            capq=math.floor(cap/fills[j]/100)*100
                            quantity=min(quantity,capq)
                        if etf and i and asset=='red' and day not in splits:
                            previous_price=float(path.iloc[i-1][asset])
                            change=fills[j]/previous_price-1
                            if (side=='buy' and change>=.095) or (side=='sell' and change<=-.095):
                                quantity=0.
                        if side=='sell':quantity=min(quantity,q[j])
                        if side=='buy':
                            affordable=max(0.,cash-5)/(fills[j]*(1+cost))
                            if lot:affordable=math.floor(affordable/100)*100
                            quantity=min(quantity,affordable)
                        if quantity<requested-1e-7:clipped+=1
                        if quantity<=1e-9:
                            order_log.append(dict(date=day,signal_date=signal_day,status='unfilled',asset=asset,requested=requested,filled=0.))
                            continue
                        notional=float(quantity*fills[j]);fee=max(5.,notional*cost)
                        direction=1 if side=='buy' else -1
                        cash-=direction*notional+fee;q[j]+=direction*quantity
                        fee_today+=fee
                        execution_pnl+=direction*quantity*(prices[j]-fills[j])
                        trades.append(dict(date=day,signal_date=signal_day,asset=asset,side=side,
                            quantity=quantity,price=fills[j],notional=notional,cost=fee,
                            capacity_limit=None if not np.isfinite(cap) else cap))
                        order_log.append(dict(date=day,signal_date=signal_day,status='filled' if quantity>=requested-1e-7 else 'partial',
                            asset=asset,requested=requested,filled=quantity))
        for event in records.get(day,[]):
            entitlements[event['event_id']]=q[1]*event['cash_per_share']
            event_log.append(dict(date=day,type='dividend_record',event_id=event['event_id'],quantity=q[1],
                                  amount=entitlements[event['event_id']]))
        receivable=float(sum(receivables.values()))
        nav=float(cash+q@prices+receivable)
        if cash< -1e-7 or q.min()< -1e-7:raise AssertionError('现金/份额为负')
        daily.append(dict(date=day,nav=nav,cash=max(cash,0.),red_quantity=q[0],bond_quantity=q[1],
            red_value=q[0]*prices[0],bond_value=q[1]*prices[1],receivable=receivable,
            trading_cost=fee_today,replication_cost=management,dividend_accrued=dividend_accrued,
            dividend_paid=dividend_paid,previous_nav=previous_nav,
            market_pnl=before_trades-previous_nav,execution_pnl=execution_pnl,clipped_orders=clipped))
        previous_nav=nav
        if day in signals.index and i+delay<len(path):
            signal=signals.loc[day]
            queue[i+delay]=(day,signal[list(ASSETS)].to_numpy(dtype=float),bool(signal.risk))
    return (pd.DataFrame(daily).set_index('date'),
        pd.DataFrame(trades,columns=['date','signal_date','asset','side','quantity','price','notional','cost','capacity_limit']),
        pd.DataFrame(event_log,columns=['date','type','event_id','quantity','quantity_before','quantity_after','amount']),
        pd.DataFrame(order_log,columns=['date','signal_date','status','asset','requested','filled']))
