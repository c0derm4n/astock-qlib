"""运行前冻结协议和输入，保留全部配置，输出只新增到独立目录。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from research.dividend_trend.run import ROOT,digest,write_json,metrics
from .data import load
from .engine import simulate
from .signals import candidates,targets
from .verify import verify

PROTOCOL=Path(__file__).with_name('protocol.json')


def todo_write(directory,phase):
    phases=['registration','replay','verification','delivery']
    pos=phases.index(phase) if phase in phases else len(phases)
    write_json(directory/'status.json',{'phase':phase,'tasks':[
        {'task':name,'status':'completed' if i<pos else 'in_progress' if i==pos else 'pending'}
        for i,name in enumerate(phases)]})


def register(data):
    data=data.resolve();manifest=data/'manifest.json'
    assert manifest.exists()
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert [c['id'] for c in candidates()]==protocol['registered_cases']
    stamp=pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory=ROOT/'output/dividend_grid/runs'/f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True,exist_ok=False)
    protected=[PROTOCOL,*sorted(Path(__file__).parent.glob('*.py')),manifest,
        ROOT/'data/formal/market.duckdb',ROOT/'strategy_registry.json',ROOT/'output/strategies/current.json',
        ROOT/'docs/INVESTMENT_PREFERENCES.md',ROOT/'strategy_corporate_actions.json',
        ROOT/'output/dividend_trend/review_20261110/completion.json',
        ROOT/'output/dividend_trend/runs/20261110-091255-521a71cf/results.json']
    write_json(directory/'registration.json',{'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'data_directory':str(data.relative_to(ROOT)),'protocol':protocol,'candidates':candidates(),
        'protected_hashes':{str(p.relative_to(ROOT)):digest(p) for p in protected},
        'run_kind':'retrospective_new_hypothesis_after_old_trend_results',
        'primary':'grid_risk_bond','production_changed':False,'investment_admission':False})
    todo_write(directory,'replay');print(str(directory))


def save_case(directory,sid,mode,candidate,signals,result,execution,benchmark):
    folder=directory/'cases'/sid;folder.mkdir(parents=True,exist_ok=False)
    daily,trades,events,orders=result
    daily.to_csv(folder/'daily.csv',index_label='date');trades.to_csv(folder/'trades.csv',index=False)
    events.to_csv(folder/'events.csv',index=False);orders.to_csv(folder/'orders.csv',index=False)
    signals.to_csv(folder/'signals.csv',index_label='date')
    stats=metrics(daily.nav)
    row={'id':sid,'candidate':candidate['id'],'mode':mode,**stats,
        'execution':execution,'trade_count':len(trades),'trading_cost':float(daily.trading_cost.sum()),
        'replication_cost':float(daily.replication_cost.sum()),'dividend_paid':float(daily.dividend_paid.sum()),
        'clipped_order_count':int(daily.clipped_orders.sum()),
        'average_red_weight':float((daily.red_value/daily.nav).mean()),
        'average_bond_weight':float((daily.bond_value/daily.nav).mean()),
        'average_cash_weight':float((daily.cash/daily.nav).mean()),
        'risk_signal_weeks':int(signals.risk.sum()),
        'grid_add_count':int(signals.reason.eq('grid_add').sum()),
        'grid_reduce_count':int(signals.reason.eq('grid_reduce').sum()),
        'dominates_passive':bool(stats['cagr']>benchmark['cagr'] and
            stats['max_drawdown']<benchmark['max_drawdown'] and stats['worst_full_year']>benchmark['worst_full_year']),
        'personal_goals':{'cagr':stats['cagr']>=.135,'worst_complete_year':stats['worst_full_year']>=-.05,
                          'max_drawdown':stats['max_drawdown']<=.40}}
    write_json(folder/'metrics.json',row)
    return row


def run(directory):
    directory=directory.resolve()
    assert not (directory/'results.json').exists(),'已完成run不得覆盖'
    registration=json.loads((directory/'registration.json').read_text(encoding='utf-8'))
    for path,expected in registration['protected_hashes'].items():
        assert digest(ROOT/path)==expected,path
    proxy,etf,actions=load(ROOT/registration['data_directory'])
    protocol=registration['protocol'];rows=[];curves={};benchmarks={}
    for mode,market,start in [('proxy',proxy[['red','bond','bond_tradeable']],protocol['start']),
                              ('etf',etf,protocol['etf_start'])]:
        raw=market.loc[start:,'red'].copy()
        if mode=='etf':raw.loc['2021-10-22':]*=2.
        passive=100000*raw/raw.iloc[0]
        benchmarks[mode]=metrics(passive)
        passive.to_csv(directory/f'{mode}_passive.csv',index_label='date')
        for candidate in registration['candidates']:
            signal=targets(proxy,candidate,start)
            settings={'etf':mode=='etf','cost':.001,'delay':1,'haircut':.006,'capacity':.05}
            sid=f'{candidate["id"]}__{mode}'
            result=simulate(market,signal,start,actions,**settings)
            row=save_case(directory,sid,mode,candidate,signal,result,settings,benchmarks[mode])
            rows.append(row);curves[sid]=result[0].nav
            if candidate['id'] in ['grid_risk_bond','grid_bond','static60_bond']:
                for name,args in [('cost2',{'cost':.002}),('delay2',{'delay':2})]:
                    config={**settings,**args};out=simulate(market,signal,start,actions,**config)
                    rows.append(save_case(directory,sid+'__'+name,mode,candidate,signal,out,config,benchmarks[mode]))
            if candidate['id']!='grid_risk_bond':continue
            end_signal=targets(proxy,candidate,start,week_end=True)
            out=simulate(market,end_signal,start,actions,**settings)
            rows.append(save_case(directory,sid+'__week_end',mode,candidate,end_signal,out,settings,benchmarks[mode]))
            extras=([('fee1pct',dict(haircut=.01)),('carry_gap',dict(carry_gap=True))] if mode=='proxy' else
                    [('no_capacity',dict(capacity=None))])
            for name,args in extras:
                config={**settings,**args};out=simulate(market,signal,start,actions,**config)
                rows.append(save_case(directory,sid+'__'+name,mode,candidate,signal,out,config,benchmarks[mode]))
            print(json.dumps({'case':sid,'cagr':row['cagr'],'worst_year':row['worst_full_year'],
                              'drawdown':row['max_drawdown']},ensure_ascii=False),flush=True)
    assert len(rows)==57,len(rows)
    segments=[]
    for sid,nav in curves.items():
        windows=([('2018-01-01','2021-12-31'),('2022-01-01','2025-12-31'),('2018-01-01','2024-12-31')]
                 if sid.endswith('__proxy') else [('2020-01-01','2024-12-31'),('2020-01-01','2025-12-31')])
        for begin,end in windows:
            before=nav.index[nav.index<begin][-1]
            segments.append({'id':sid,'window':begin[:4]+'-'+end[:4],**metrics(nav.loc[before:end])})
    todo_write(directory,'verification')
    checks=verify(directory,proxy,etf,actions,rows,registration)
    write_json(directory/'results.json',{'protocol':protocol['version'],'primary':registration['primary'],
        'benchmarks':benchmarks,'rows':rows,'segments':segments,'case_count':len(rows),
        'production_changed':False,'investment_admission':False})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('annual','execution','personal_goals')}
                  for r in rows]).to_csv(directory/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'id':r['id'],**a} for r in rows for a in r['annual']]).to_csv(
        directory/'annual_returns.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{k:v for k,v in r.items() if k!='annual'} for r in segments]).to_csv(
        directory/'segments.csv',index=False,encoding='utf-8-sig')
    todo_write(directory,'delivery')
    print(json.dumps({'run':str(directory),'verification':checks,
        'main_rows':[{k:v for k,v in r.items() if k in ('id','final_nav','cagr','max_drawdown',
                    'worst_full_year','worst_year','trade_count','clipped_order_count','dominates_passive')}
                    for r in rows if r['candidate'] in protocol['registered_cases'][:9] and r['id'].count('__')==1]},
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',type=Path);parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.register:register(args.register)
    elif args.run:run(args.run)
    else:parser.error('需要--register <数据目录> 或 --run <登记目录>')
