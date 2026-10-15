"""预登记的周频跨行业与固定通信同现金账本对照；不修改旧研究。"""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import universe
from src.artifacts import write_json_atomic
from src.portfolio import risk_group
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import corrected_market, read, ROOT, SOURCE
from research.goal30.primary_cash_ledger import EventCalendar, new_state, settle_day, freeze_next, validate_state
from research.goal30.daily_cash_ledger import seal
from research.goal30.validate_primary_execution import PrimaryMarketDays

PROTOCOL = Path(__file__).with_name('dynamic_protocol.json')
PRIOR = BASE/'output/goal30/dividend_payment_replay/20260914-083340-83750109'
OUT = BASE/'output/two_line/dynamic_industry'
GOLD, COMM, CSI = 'SH518880', 'SH515880', 'SH510300'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def equity(symbol):
    return universe.get_asset_class(symbol[2:]) == 'equity'

def features(prices):
    valid = np.isfinite(prices) & prices.gt(0)
    score = (prices/prices.shift(60)-1).where(valid.rolling(61).sum().eq(61))
    trend = valid.rolling(200).sum().eq(200) & prices.gt(prices.rolling(200).mean())
    return score, trend

def targets_for(variant, prices, allowed, current, holding_days, exiting):
    scores, trends = features(prices)
    score, trend = scores.iloc[-1], trends.iloc[-1]
    candidates = {s: float(v) for s,v in score.items() if equity(s) and s in allowed and trend[s]
                  and s not in exiting and np.isfinite(v)}
    rank = pd.Series(candidates, dtype=float).rank(ascending=False, method='min').to_dict()
    chosen=[]
    if variant.startswith('fixed'):
        symbol = COMM if COMM in allowed else CSI
        raw={symbol:.6} if symbol in allowed and trend.get(symbol,False) and symbol not in exiting else {}
    else:
        held=[s for s in current if s in candidates and current[s]>0]
        removable=sorted([s for s in held if rank[s]>5 and holding_days.get(s,0)>=20],
                          key=lambda s:(candidates[s],s))[:1]
        chosen=[s for s in held if s not in removable]
        used={risk_group(s) for s in chosen}
        for value in sorted(set(candidates.values()),reverse=True):
            group=[s for s,v in candidates.items() if v==value and s not in chosen and s not in removable
                   and risk_group(s) not in used]
            if not group:
                continue
            if len(group)>3-len(chosen):
                break
            if len({risk_group(s) for s in group})!=len(group):
                continue
            chosen+=group
            used.update(risk_group(s) for s in group)
            if len(chosen)==3:
                break
        raw={s:.2 for s in chosen}
    if GOLD in allowed and trend.get(GOLD,False) and GOLD not in exiting:
        raw[GOLD]=.35
    returns=prices.pct_change(fill_method=None).tail(60)
    raw={s:w for s,w in raw.items() if len(returns)==60 and np.isfinite(returns[s]).all()}
    volatility=0.
    if raw:
        symbols=list(raw)
        weights=np.array(list(raw.values()))
        cov=returns[symbols].cov().to_numpy()*252
        volatility=float(np.sqrt(max(0.,weights@cov@weights)))
        scale=min(1.,.15/max(volatility,1e-15))
        raw={s:w*scale for s,w in raw.items()}
    require(sum(raw.values())<=.950000001,'总目标超限')
    if not variant.startswith('fixed'):
        eq=[s for s in raw if equity(s)]
        require(len(eq)<=3 and len({risk_group(s) for s in eq})==len(eq),'分散失效')
    return raw, {'scores':candidates,'ranks':rank,'chosen':chosen,'raw_volatility':volatility,
                 'target_volatility':min(volatility,.15)}

def checks():
    dates=pd.bdate_range('2020-01-01',periods=260)
    symbols=['SH515880','SH512720','SH512800','SH512010',GOLD]
    p=pd.DataFrame({s:100*np.exp(np.arange(260)*(.001+i*.0001)) for i,s in enumerate(symbols)},index=dates)
    a=targets_for('dynamic',p.iloc[:230],set(symbols),{}, {},[])
    altered=p.copy();altered.iloc[230:]*=100
    require(a==targets_for('dynamic',altered.iloc[:230],set(symbols),{},{},[]),'未来数据影响过去')
    missing=p.copy();missing.loc[dates[100],symbols[0]]=np.nan
    require(symbols[0] not in targets_for('dynamic',missing,set(symbols),{},{},[])[0],'缺数据错误回填')
    require(targets_for('dynamic',p.iloc[:199],set(symbols),{},{},[])[0]=={},'预热不足买入')
    target=a[0]
    require(sum(target.values())<=.950000001 and sum(w for s,w in target.items() if equity(s))<=.600000001,'现金/权益约束')
    require(all(w<=.200000001 for s,w in target.items() if equity(s)),'单权益约束')
    fixed=targets_for('fixed',p,set(symbols),{},{},[])[0]
    require(CSI not in fixed,'合格通信不该切沪深300')
    tie_symbols=['SH515880','SH512800','SH512010','SH512400',GOLD]
    tie=pd.DataFrame({s:p.iloc[:,0].values for s in tie_symbols},index=dates)
    tie[GOLD]=p[GOLD]
    tied=targets_for('dynamic',tie,set(tie_symbols),{},{},[])[0]
    require(not any(equity(s) for s in tied),'边界并列必须留现金，不买低分候选')
    # Two tied leaders exceed the single free slot; a lower-scoring third may not fill it.
    from unittest.mock import patch
    mock_score=pd.DataFrame([{s:v for s,v in zip(tie_symbols,[.1,.1,.3,.4,.0])}],index=[dates[-1]])
    mock_trend=pd.DataFrame(True,index=[dates[-1]],columns=tie_symbols)
    with patch(__name__+'.features',return_value=(mock_score,mock_trend)):
        boundary=targets_for('dynamic',tie,set(tie_symbols),{'SH512010':.2,'SH512400':.2},
                             {'SH512010':1,'SH512400':1},[])[0]
    require(set(s for s in boundary if equity(s))=={'SH512010','SH512400'},'已有两仓的边界并列错误填仓')
    return {'passed':True,'checks':['future_prefix','no_price_fill','200day_warmup','equity_groups','cash_reserve','single_equity_cap','fixed_proxy','boundary_tie_cash','held_boundary_tie_cash']}

def register():
    spec=read(PROTOCOL);old=read(PRIOR/'registration.json')
    directory=OUT/(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    policy={**old['policy'],'max_weight':.6,'max_equity_weight':.6,'max_group_weight':.6,
            'max_gross_weight':.95,'max_daily_turnover':.3,'conservative_buy_budget':True}
    paths=[PROTOCOL,Path(__file__),PRIOR/'registration.json',ROOT/'signal_series.parquet',ROOT/'actions.json',
           SOURCE/'market_snapshot.parquet',BASE/'research/goal30/primary_cash_ledger.py',
           BASE/'research/goal30/daily_cash_ledger.py',BASE/'src/portfolio.py']
    # Current source hashes record provenance without rewriting historical hashes.
    inputs={p.relative_to(BASE).as_posix():digest(p) for p in paths}
    for path in old.get('input_hashes',{}):
        p=BASE/path
        if p.exists() and (path.endswith('.parquet') or path.endswith('actions.json')):
            inputs[path]=digest(p)
    reg={'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'protocol':spec,'policy':policy,
         'actions':old['actions'],'input_hashes':inputs,'small_checks':checks(),
         'investment_admission':False,'production_pointer_changed':False}
    write_json_atomic(directory/'registration.json',reg)
    shutil.copyfile(PROTOCOL,directory/PROTOCOL.name);shutil.copyfile(__file__,directory/Path(__file__).name)
    print(json.dumps({'registered':str(directory),'checks':reg['small_checks']}),flush=True)
    return directory

def drive(frames,prices,reg,variant,scenario,dates=None):
    spec=reg['protocol'];policy=reg['policy']
    dates=frames.days[(frames.days>=spec['start'])&(frames.days<=spec['end'])] if dates is None else dates
    state=new_state(policy['initial_capital'],frames.context(dates[0])['previous_date'])
    calendar=EventCalendar(reg['actions'],frames.days,payment_mode='announced_schedule_model',
                           uncertain_payment_extra_sessions=1)
    data={k:[] for k in ('report','trades','actions','holdings','signals','orders','state_hashes')}
    data.update(pending={},issue_totals={})
    queued=[];last_week=None
    for i,day in enumerate(dates):
        context=frames.context(day)
        # A delayed order is released unchanged; internal ledger submission day is separately disclosed.
        ready=[q for q in queued if q['execution_date']==context['date']]
        queued=[q for q in queued if q['execution_date']!=context['date']]
        for q in ready:
            q['signal_date']=state['last_date']
        state['pending_orders']=ready
        state['next_execution_date']=context['date'] if ready else None
        state=seal(state);data['pending'][context['date']]=copy.deepcopy(ready)
        state,output=settle_day(state,context,policy,calendar,slippage_multiplier=scenario['slippage_multiplier'])
        event_symbols={e['symbol'] for phase,events in calendar.for_day(day).items() if phase in ('split','ex') for e in events}
        cancelled=[q for q in queued if q['symbol'] in event_symbols]
        queued=[q for q in queued if q['symbol'] not in event_symbols]
        if cancelled:
            data['issue_totals']['delayed_action_cancelled']=data['issue_totals'].get('delayed_action_cancelled',0)+len(cancelled)
        origin={q['symbol']:q['original_signal_date'] for q in ready}
        for trade in output['trades']:
            trade['submission_date']=trade['signal_date'];trade['signal_date']=origin[trade['symbol']]
        for kind in ('report','trades','actions','holdings'):
            data[kind].extend([output[kind]] if kind=='report' else output[kind])
        for issue,count in output['issues'].items():
            data['issue_totals'][issue]=data['issue_totals'].get(issue,0)+count
        week=str(day.to_period('W-SUN'))
        decision_day=week!=last_week
        if decision_day:
            last_week=week
            allowed=set(frames.eligible.columns[frames.eligible.loc[day].fillna(False)])
            current={h['symbol']:h['value']/output['report']['account'] for h in output['holdings']}
            holding={s:state['sequence']-b+1 for s,b in state['bought'].items()}
            target,trace=targets_for(variant,prices.loc[:day],allowed,current,holding,state['exiting'])
            data['signals'].append({'date':context['date'],'targets':target,'current':current,**trace})
            offset=scenario['execution_delay']
            if i+offset<len(dates):
                temp=freeze_next(state,target,context['close'],frames.adv.loc[day].to_dict(),policy,dates[i+offset])
                for order in temp['pending_orders']:
                    order['original_signal_date']=context['date']
                queued+=temp['pending_orders']
                data['orders'].extend(copy.deepcopy(temp['pending_orders']))
        state=json.loads(json.dumps(state,allow_nan=False));validate_state(state)
        data['state_hashes'].append({'date':context['date'],'state_sha256':state['state_sha256']})
    data['final_state']=state
    return data

def metrics(data):
    report=pd.DataFrame(data['report']).set_index('date');report.index=pd.to_datetime(report.index)
    nav=report.account;returns=nav.pct_change(fill_method=None);returns.iloc[0]=nav.iloc[0]/1e6-1
    annual=(1+returns).groupby(returns.index.year).prod()-1
    complete=annual[(annual.index>nav.index[0].year)&(annual.index<nav.index[-1].year)]
    cagr=float((nav.iloc[-1]/1e6)**(365.25/(nav.index[-1]-nav.index[0]).days)-1)
    mdd=float(-(nav/np.maximum.accumulate(np.r_[1e6,nav.values])[1:]-1).min())
    worst=float(complete.min())
    end2024=nav.loc[:'2024-12-31']
    before=float((end2024.iloc[-1]/1e6)**(365.25/(end2024.index[-1]-nav.index[0]).days)-1)
    ex=returns[returns.index.year!=2025]
    holds=pd.DataFrame(data['holdings'])
    dependencies={}
    if len(holds):
        holds['weight']=holds['value']/holds['date'].map({str(d.date()):float(v) for d,v in nav.items()})
        dependencies={s:float(v/len(nav)) for s,v in holds.groupby('symbol').weight.sum().items()}
    return {'cagr':cagr,'max_drawdown':mdd,'worst_complete_year':worst,'worst_year':int(complete.idxmin()),
            'annual_returns':{str(y):float(v) for y,v in annual.items()},'through_2024_cagr':before,
            'excluding_2025_compounded_return':float((1+ex).prod()-1),
            'excluding_2025_annualized_252_sessions':float((1+ex).prod()**(252/len(ex))-1),
            'mean_symbol_weights':dependencies,'traded_symbols':sorted({t['symbol'] for t in data['trades']}),
            'goals':{'cagr':cagr>=.135,'annual':worst>=-.05,'drawdown':mdd<=.4,
                     'joint':cagr>=.135 and worst>=-.05 and mdd<=.4}}

def audit(data,frames,scenario):
    dates=pd.DatetimeIndex([r['date'] for r in data['report']])
    expected=dates[~dates.to_period('W-SUN').duplicated()]
    actual=pd.DatetimeIndex([s['date'] for s in data['signals']])
    require(expected.equals(actual),'非周首决策')
    for trade in data['trades']:
        require(frames.days.get_loc(pd.Timestamp(trade['date']))-frames.days.get_loc(pd.Timestamp(trade['signal_date']))==scenario['execution_delay'],'成交延迟错误')
    max_error=0.
    cash=1e6
    trades=pd.DataFrame(data['trades']);actions=data['actions']
    for row in data['report']:
        day=row['date']
        for t in data['trades']:
            if t['date']==day:
                cash+=(1 if t['side']=='sell' else -1)*t['quantity']*t['price']-t['commission']
        cash+=sum(a['amount'] for a in actions if a['date']==day and a['type']=='pay_dividend')
        max_error=max(max_error,abs(cash-row['cash']))
        require(cash>=-1e-6 and abs(cash-row['cash'])<1e-5,'现金不能独立对账')
        require(abs(row['account']-row['cash']-row['value']-row['dividend_receivable'])<1e-5,'净值不平')
    return {'weekly_decisions':len(expected),'trades':len(data['trades']),'cash_max_error':max_error,'passed':True}

def run(directory):
    reg=read(directory/'registration.json')
    for p,h in reg['input_hashes'].items(): require(digest(BASE/p)==h,'登记输入改变:'+p)
    frames=PrimaryMarketDays(corrected_market());prices=frames.market.wide('adj_close')
    require(sum(equity(s) for s in prices)==27,'权益池不是27只')
    dates=frames.days[(frames.days>=reg['protocol']['start'])&(frames.days<=reg['protocol']['end'])]
    smoke=drive(frames,prices,reg,reg['protocol']['variants'][1],reg['protocol']['scenarios'][2],dates[:35])
    smoke_check=audit(smoke,frames,reg['protocol']['scenarios'][2]);write_json_atomic(directory/'smoke.json',smoke_check)
    results={'cases':{},'investment_admission':False,'history_is_independent_holdout':False}
    for variant in reg['protocol']['variants']:
        for scenario in reg['protocol']['scenarios']:
            key=variant+'__'+scenario['name']
            data=drive(frames,prices,reg,variant,scenario)
            item={**metrics(data),'audit':audit(data,frames,scenario),'issues':data['issue_totals']}
            write_json_atomic(directory/(key+'.json'),data)
            for kind in ('report','trades','holdings','signals','orders'):
                pd.DataFrame(data[kind]).to_csv(directory/(key+'_'+kind+'.csv'),index=False)
            results['cases'][key]=item;write_json_atomic(directory/'results.json',results)
            print(json.dumps({'case':key,'cagr':item['cagr'],'mdd':item['max_drawdown'],'worst':item['worst_complete_year'],'audit':item['audit']}),flush=True)
    write_json_atomic(directory/'completion.json',{'cases':len(results['cases']),
        'registration_sha256':digest(directory/'registration.json'),'results_sha256':digest(directory/'results.json')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','register','run']);parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.mode=='check': print(json.dumps(checks()))
    elif args.mode=='register': register()
    else: run(args.directory)
