"""风险优先保留的12%版本；明确记录已见压力结果后的选择，独立登记再验证。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pandas as pd

from .run import ROOT, digest, write_json, todo_write, metrics, replay, save_case, verify_cases
from .audit import independent_targets, independent_units_check

SID = 'dual_blend_ma200_v12'
SCENARIOS = {'base': {}, 'cost2': {'cost':.002}, 'cost3': {'cost':.003},
    'delay2': {'delay':2}, 'week_end': {'week_end':True}, 'fee1pct': {'haircut':.01},
    'ma150': {'ma':150}, 'ma250': {'ma':250}, 'vol10': {'vol':.10}, 'vol14': {'vol':.14}}


def register(source, validation):
    source, validation = source.resolve(), validation.resolve()
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT/'output/dividend_trend/runs'/f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True,exist_ok=False)
    oldreg = json.loads((source/'registration.json').read_text(encoding='utf-8'))
    files = [Path(__file__), ROOT/'research/dividend_trend/audit.py',
             validation/'registration.json', validation/'results.json', validation/'verification.json']
    hashes = {**oldreg['hashes'],**{str(p.relative_to(ROOT)):digest(p) for p in files}}
    write_json(directory/'registration.json',{
        'candidate':SID,'source_run':str(source.relative_to(ROOT)),
        'prior_validation':str(validation.relative_to(ROOT)),
        'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'known_at_selection':{'v15_cagr':.10282486318297668,'v15_drawdown':.2128152564459861,
            'v15_worst_year':-.057198573550799936,'v12_cagr':.10170418867977227,
            'v12_drawdown':.1746039784118819,'v12_worst_year':-.03741027627579596},
        'selection_reason':'12%预算在已见参数压力中，两项风险均改善，年化略降；按用户风险优先保留补充候选，不更换v2已登记126日主方案，不声称盲测或未见历史。',
        'protocol':{**oldreg['protocol'],'version':'dividend-relative-risk12-20261110-v3',
            'primary_candidate':SID,'vol_budget':.12},
        'scenarios':SCENARIOS,'hashes':hashes,'candidates':[],
        'sample_kind':'post_selection_historical_research','future_unseen_holdout':False,
        'production_changed':False,'investment_admission':False})
    todo_write(directory,'replay')
    print(str(directory))


def run(directory):
    directory=directory.resolve()
    assert not (directory/'results.json').exists()
    reg=json.loads((directory/'registration.json').read_text(encoding='utf-8'))
    for path,expected in reg['hashes'].items():
        assert digest(ROOT/path)==expected,path
    source=ROOT/reg['source_run']
    original=json.loads((source/'results.json').read_text(encoding='utf-8'))
    prices=pd.read_csv(source/'index_gold_prices.csv',parse_dates=['date']).set_index('date')
    prices.index=pd.DatetimeIndex(prices.index).as_unit('ns')
    etf=pd.read_csv(source/'etf_gold_prices.csv',parse_dates=['date']).set_index('date')
    etf.index=pd.DatetimeIndex(etf.index).as_unit('ns')
    rows,paths=[],{}
    for name,arguments in reg['scenarios'].items():
        settings={'ma':200,'vol':.12}
        settings.update({k:v for k,v in arguments.items() if k in settings})
        fills={k:v for k,v in arguments.items() if k not in settings}
        targets=independent_targets(prices,**settings)
        daily,trades=replay(prices,targets,'2016-09-12',**fills)
        case=SID if name=='base' else f'{SID}__{name}'
        rows.append(save_case(directory,case,daily,trades,original['benchmark']))
        paths[case]=daily.nav
    targets=independent_targets(prices,vol=.12)
    for name,args in {'base':{},'cost2':{'cost':.002},'delay2':{'delay':2}}.items():
        daily,trades=replay(etf,targets,'2019-01-18',haircut=0,
                            no_fill_dates=['2021-10-22'],**args)
        rows.append(save_case(directory,f'{SID}__etf_{name}',daily,trades,original['etf_benchmark']))
    todo_write(directory,'validation')
    verification=verify_cases(directory,prices,reg,rows)
    verification['index_unit_reconstruction_max_error']=independent_units_check(prices,directory/'cases'/SID,.006)
    verification['etf_unit_reconstruction_max_error']=independent_units_check(etf,directory/'cases'/f'{SID}__etf_base',0.)
    prior=json.loads((ROOT/reg['prior_validation']/'results.json').read_text(encoding='utf-8'))
    priorrow=next(r for r in prior['rows'] if r['id'].endswith('__vol12'))
    for field in ('final_nav','cagr','max_drawdown','worst_full_year'):
        assert abs(rows[0][field]-priorrow[field])<1e-9,field
    prefix=prices.loc[:'2021-12-31']
    prefix_targets=independent_targets(prefix,vol=.12)
    pd.testing.assert_frame_equal(prefix_targets,targets.loc[prefix.index])
    daily,_=replay(prefix,prefix_targets,'2016-09-12')
    prefixerror=float((daily.nav-paths[SID].loc[daily.index]).abs().max())
    assert prefixerror<1e-7
    verification['causal_prefix_max_error']=prefixerror
    segments=[]
    nav=paths[SID]
    for begin,end in [('2017-01-01','2021-12-31'),('2022-01-01','2025-12-31'),('2017-01-01','2024-12-31')]:
        before=nav.index[nav.index<begin][-1]
        segments.append({'window':begin[:4]+'-'+end[:4],**metrics(nav.loc[before:end])})
    write_json(directory/'verification.json',verification)
    write_json(directory/'results.json',{'candidate':SID,'source_run':reg['source_run'],
        'rows':rows,'segments':segments,'benchmark':original['benchmark'],
        'etf_benchmark':original['etf_benchmark'],'production_changed':False,'investment_admission':False})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('annual','personal_goals')}
                  for r in rows]).to_csv(directory/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'id':r['id'],**a} for r in rows for a in r['annual']]).to_csv(
        directory/'annual_returns.csv',index=False,encoding='utf-8-sig')
    todo_write(directory,'delivery')
    print(json.dumps({'directory':str(directory),'verification':verification,
        'rows':[{k:v for k,v in r.items() if k in ('id','final_nav','cagr','max_drawdown',
             'worst_full_year','worst_year','dominates_red_benchmark')} for r in rows]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.register and args.validation:
        register(args.register,args.validation)
    elif args.run:
        run(args.run)
    else:
        parser.error('需要--register <来源> --validation <已有验证>或--run <目录>')
