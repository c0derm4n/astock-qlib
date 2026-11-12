"""不调用交易引擎重建现金、份额、应收与净值，独立核对指标和因果前缀。"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from research.dividend_trend.run import ROOT,digest,write_json
from research.valuation_bands.data import build_calendar
from .data import frame_read
from .signals import targets


def independent_metrics(nav,row):
    peak=float(nav.iloc[0]);dd=0.
    for value in nav:
        peak=max(peak,float(value));dd=max(dd,(peak-value)/peak)
    ye=nav.groupby(nav.index.year).last()
    full=ye.pct_change(fill_method=None).loc[2020 if nav.index[0].year==2019 else 2017:2025]
    cagr=math.expm1(math.log(nav.iloc[-1]/nav.iloc[0])*365.2425/(nav.index[-1]-nav.index[0]).days)
    error=max(abs(cagr-row['cagr']),abs(dd-row['max_drawdown']),abs(float(full.min())-row['worst_full_year']))
    assert error<1e-12,error
    return error


def reconstruct(folder,market,actions,config):
    daily=frame_read(folder/'daily.csv')
    trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','signal_date'])
    market=market.loc[daily.index].copy()
    if config.get('carry_gap') and pd.Timestamp('2018-08-17') in market.index:
        pos=market.index.get_loc('2018-08-17');market.loc['2018-08-17','bond']=market.bond.iloc[pos-1]
    groups={d:f for d,f in trades.groupby('date')}
    cash=100000.;q=np.zeros(2);entitled={};receivable={};values=[]
    event_by_record={pd.Timestamp(e['record_date']):e for e in actions['bond']} if config['etf'] else {}
    event_by_ex={pd.Timestamp(e['ex_date']):e for e in actions['bond']} if config['etf'] else {}
    event_by_pay={pd.Timestamp(e['payment_date']):e for e in actions['bond']} if config['etf'] else {}
    for i,day in enumerate(daily.index):
        prices=market.loc[day,['red','bond']].to_numpy(dtype=float)
        if config['etf'] and day==pd.Timestamp('2021-10-22'):q[0]*=2
        if day in event_by_ex:
            event=event_by_ex[day];receivable[event['event_id']]=entitled.get(event['event_id'],0.)
        if day in event_by_pay:
            cash+=receivable.pop(event_by_pay[day]['event_id'],0.)
        if not config['etf'] and i:
            q[0]*=math.exp(-config.get('haircut',.006)*(day-daily.index[i-1]).days/365.2425)
        for trade in groups.get(day,pd.DataFrame()).itertuples():
            j=['red','bond'].index(trade.asset);direction=1 if trade.side=='buy' else -1
            assert trade.signal_date<trade.date
            assert abs(trade.notional-trade.quantity*trade.price)<1e-7
            assert abs(trade.cost-max(5.,trade.notional*config.get('cost',.001)))<1e-7
            q[j]+=direction*trade.quantity;cash-=direction*trade.notional+trade.cost
            if config['etf'] or trade.asset=='bond':
                assert bool(market.loc[day,trade.asset+'_tradeable'])
            else:
                # 全收益指数有逐交易日点位，代理模式没有ETF可成交标志列。
                assert np.isfinite(market.loc[day,'red']) and market.loc[day,'red']>0
            if config['etf']:
                assert abs(trade.quantity/100-round(trade.quantity/100))<1e-9
                if config.get('capacity',.05) is not None:
                    pos=market.index.get_loc(day)
                    cap=min(float(market.iloc[pos-1][trade.asset+'_adv']),float(market.loc[day,trade.asset+'_amount']))*config.get('capacity',.05)
                    assert trade.notional<=cap+1e-7
        if day in event_by_record:
            e=event_by_record[day];entitled[e['event_id']]=q[1]*e['cash_per_share']
        assert cash>=-1e-7 and q.min()>=-1e-7
        values.append(cash+float(q@prices)+sum(receivable.values()))
        assert abs(cash-daily.loc[day,'cash'])<1e-6
        assert abs(sum(receivable.values())-daily.loc[day,'receivable'])<1e-6
    error=float(np.max(np.abs(np.asarray(values)-daily.nav.to_numpy())))
    assert error<1e-6,error
    return error


def verify(directory,proxy,etf,actions,rows,registration):
    maximum_ledger_error=maximum_metric_error=0.
    for row in rows:
        folder=directory/'cases'/row['id']
        daily=frame_read(folder/'daily.csv')
        assert daily.index.equals(build_calendar(daily.index[0],daily.index[-1]))
        equation=daily.previous_nav+daily.market_pnl+daily.execution_pnl-daily.trading_cost-daily.replication_cost
        assert np.max(np.abs(equation-daily.nav))<1e-7
        assert np.allclose(daily.cash+daily.red_value+daily.bond_value+daily.receivable,daily.nav,rtol=0,atol=1e-7)
        maximum_metric_error=max(maximum_metric_error,independent_metrics(daily.nav,row))
        maximum_ledger_error=max(maximum_ledger_error,reconstruct(folder,etf if row['mode']=='etf' else proxy,
                                                               actions,row['execution']))
    candidate=next(c for c in registration['candidates'] if c['id']=='grid_risk_bond')
    prefix=proxy.loc[:'2021-12-31']
    for start in ('2016-09-12','2019-01-18'):
        pd.testing.assert_frame_equal(targets(prefix,candidate,start),targets(proxy,candidate,start).loc[:prefix.index[-1]])
    # 用numpy逐窗计算主方案全部MA和波动，独立于pandas滚动实现。
    full=targets(proxy,candidate,'2016-09-12')
    p=proxy.red.to_numpy();max_indicator_error=0.
    for day,s in full.iterrows():
        i=proxy.index.get_loc(day)
        ma200=float(np.mean(p[i-199:i+1]));ma250=float(np.mean(p[i-249:i+1]))
        ret=p[i-59:i+1]/p[i-60:i]-1
        vol=float(np.std(ret,ddof=1)*math.sqrt(252))
        max_indicator_error=max(max_indicator_error,abs(ma200-s.ma_grid),abs(ma250-s.ma_risk),abs(vol-s.vol))
    assert max_indicator_error<1e-8
    changes=[path for path,expected in registration['protected_hashes'].items() if digest(ROOT/path)!=expected]
    assert not changes,changes
    result={'status':'passed','case_count':len(rows),'independent_ledger_max_error':maximum_ledger_error,
        'independent_metric_max_error':maximum_metric_error,'independent_indicator_max_error':max_indicator_error,
        'causal_signal_prefix':True,'calendar_complete':True,'etf_round_lots_and_capacity_checked':True,
        'dividend_receivable_and_payment_checked':True,'protected_files_unchanged':True,
        'real_broker_fills_and_unseen_holdout':False,'investment_admission':False}
    write_json(directory/'verification.json',result)
    return result
