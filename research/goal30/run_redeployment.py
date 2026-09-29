"""预登记、执行并独立复核15个受限资金重新分配情景。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_cash_ledger import EventCalendar, new_state, settle_day, freeze_next, validate_state
from research.goal30.primary_research_series import START, END, corrected_market, read
from research.goal30.redeployment import decision
from research.goal30.reconcile_redeployment import verify_signals
from research.goal30.run_group_diversification import feature_bank, metrics_for, check_inputs, decision_summary
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import PrimaryMarketDays, REFERENCE, inspect_case

OUT=BASE/'output/goal30/redeployment'
PRIOR=BASE/'output/goal30/dividend_payment_replay/20260914-083340-83750109'
CODE=('redeployment.py','reconcile_redeployment.py','test_redeployment.py','run_redeployment.py')


def register():
    old=read(PRIOR/'registration.json');proof=read(PRIOR/'verification.json')
    last=read(BASE/'output/goal30/verification/remaining_dividends/requirements_audit.json')
    require(last['stage_completed'] and proof['cases_verified']==111,'上一轮未验收')
    inputs={**last['input_and_artifact_hashes'],**old['input_hashes'],**proof['artifact_hashes']}
    for p in [PRIOR/'verification.json',BASE/'output/goal30/verification/remaining_dividends/requirements_audit.json',
              *[Path(__file__).with_name(n) for n in CODE]]: inputs[p.relative_to(BASE).as_posix()]=digest(p)
    registration={'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'input_hashes':inputs,
        'actions':old['actions'],'policy':old['policy'],'variants':old['variants'],'scenarios':old['scenarios'],
        'protocol':{'start':START,'end':END,'case_count':15,'primary':'group4_risk_parity',
          'hypothesis':'Reallocate capital clipped by position/group limits within the original raw target budget, retaining the same risk and selection rules. This may reduce idle capital without adding instruments or changing entry timing.',
          'allocation':'Hierarchical capped proportional allocation: distribute raw total (capped at95%) across original risk groups with capacity=min70%,member_count*35%; then allocate proportionally within each group with35% ETF cap. Insufficient capacity remains cash. Daily removed targets and monthly unfilled slots are not restored.',
          'risk':'Apply original25% diagonal-shrunk63-return covariance and15% desired-target forecast volatility cap; retain original3pp drift, second cap and30% shared turnover budget. Separately measure forecast risk after the turnover adjustment; executed positions can differ from targets.',
          'comparison':'All five prior variants under all three prior scenarios, paired with the already verified updated-dividend-policy archive. No post-result selection or parameter tuning.',
          'verification':'Every daily account independently reconciled. Independent bisection for capacity allocation and direct covariance computation for targets, using independently recalculated full-history score prefixes.',
          'gate':'CAGR>=CNY SPY total return; daily mean drawdown<=SPY; MDD<=20%. Primary all stress and at least2/3 primary3/4/5-slot base neighborhood must pass before further review. Historical reused research is not out-of-sample or real forward.',
          'promotion':'No live pointer or daily candidate change; directory completeness, actual account reconciliation and126 real-forward sessions still required.'}}
    check_inputs(registration)
    directory=OUT/(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False);(directory/'registered_code').mkdir()
    for name in CODE: shutil.copyfile(Path(__file__).with_name(name),directory/'registered_code'/name)
    plan=BASE/'docs/SP500_REDEPLOYMENT_PLAN.md';shutil.copyfile(plan,directory/'registered_plan.md')
    registration['registered_plan_sha256']=digest(directory/'registered_plan.md')
    write_json_atomic(directory/'registration.json',registration)
    write_json_atomic(OUT/'latest_registration.json',{'directory':directory.relative_to(BASE).as_posix()})
    print(json.dumps({'registered':directory.relative_to(BASE).as_posix(),'cases':15,'inputs':len(inputs)}),flush=True)


def drive(frames,prices,features,registered,variant,scenario):
    dates=frames.days[(frames.days>=START)&(frames.days<=END)];policy=registered['policy']
    state=new_state(policy['initial_capital'],frames.context(dates[0])['previous_date'])
    calendar=EventCalendar(registered['actions'],frames.days,payment_mode='announced_schedule_model',
        payment_delay_sessions=scenario['payment_delay_sessions'],uncertain_payment_extra_sessions=scenario['uncertain_payment_extra_sessions'])
    selection={};data={'report':[],'trades':[],'actions':[],'holdings':[],'signals':[],'state_hashes':[],'pending':{},'issue_totals':{}}
    for i,day in enumerate(dates):
        context=frames.context(day);data['pending'][context['date']]=state['pending_orders']
        state,output=settle_day(state,context,policy,calendar,slippage_multiplier=scenario['slippage_multiplier'])
        data['report'].append(output['report'])
        for kind in ('trades','actions','holdings'): data[kind].extend(output[kind])
        for issue,count in output['issues'].items(): data['issue_totals'][issue]=data['issue_totals'].get(issue,0)+count
        if i+1<len(dates):
            allowed=frames.eligible.columns[frames.eligible.loc[day].fillna(False)].tolist()
            current={h['symbol']:h['value']/(output['report']['cash']+output['report']['value']) for h in output['holdings']}
            selection,targets,trace=decision(selection,features.loc[day].dropna().to_dict(),prices.loc[:day],allowed,state['exiting'],current,policy,variant)
            data['signals'].append({'date':context['date'],'eligible':sorted(allowed),'exiting':state['exiting'],
                'current_weights':current,'selection_state':selection,'target_weights':targets,**trace})
            state=freeze_next(state,targets,context['close'],frames.adv.loc[day].to_dict(),policy,dates[i+1])
        state=json.loads(json.dumps(state,allow_nan=False));validate_state(state)
        data['state_hashes'].append({'date':context['date'],'state_sha256':state['state_sha256']})
    data['final_state']=state
    return data


def paired_metrics(item,prior):
    return {'cagr_delta':item['metrics']['strategy']['cagr']-prior['metrics']['strategy']['cagr'],
        'mean_daily_drawdown_delta':item['metrics']['strategy']['mean_daily_drawdown']-prior['metrics']['strategy']['mean_daily_drawdown'],
        'max_drawdown_delta':item['metrics']['strategy']['max_drawdown']-prior['metrics']['strategy']['max_drawdown'],
        'mean_invested_fraction_delta':item['mean_invested_fraction']-prior['mean_invested_fraction'],
        'trade_count_delta':item['trade_count']-prior['trade_count'],
        'two_way_turnover_delta':item['sum_daily_two_way_turnover']-prior['sum_daily_two_way_turnover']}


def run(directory):
    reg=read(directory/'registration.json');check_inputs(reg)
    with (directory/'started.json').open('x',encoding='utf-8') as stream: json.dump({'started_at':pd.Timestamp.now().isoformat()},stream)
    result={'cases':{},'investment_admission':False,'full_goal_achieved':False}
    try:
        frames=PrimaryMarketDays(corrected_market());prices,features,bank,feature_proof=feature_bank(frames)
        result['feature_verification']=feature_proof
        benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True)
        prior=read(PRIOR/'results.json')['cases']
        for variant in reg['variants']:
            for scenario in reg['scenarios']:
                key=variant['name']+'__'+scenario['name'];started=time.monotonic()
                data=drive(frames,prices,features,reg,variant,scenario)
                item={**metrics_for(data,benchmark),'variant':variant,'scenario':scenario,
                    'ledger':inspect_case(frames,reg,scenario,reg['actions'],data),
                    'signals':verify_signals(frames,prices,bank,reg,variant,data),'issues':data['issue_totals']}
                item['paired_comparison']=paired_metrics(item,prior[key])
                path=directory/(key+'.json');write_json_atomic(path,data)
                pd.DataFrame(data['report']).to_csv(directory/(key+'_report.csv'),index=False)
                item.update(artifact=path.name,sha256=digest(path),seconds=time.monotonic()-started)
                result['cases'][key]=item;write_json_atomic(directory/'results.json',result)
                print(json.dumps({'case':key,'cagr':item['metrics']['strategy']['cagr'],
                    'mean_drawdown':item['metrics']['strategy']['mean_daily_drawdown'],'max_drawdown':item['metrics']['strategy']['max_drawdown'],
                    'invested':item['mean_invested_fraction'],'paired':item['paired_comparison'],
                    'post_budget_risk_breaches':item['signals']['post_budget_target_volatility_above_15pct_days'],'seconds':item['seconds']}),flush=True)
        require(len(result['cases'])==15,'情景未全部完成');result['selection_gate']=decision_summary(result['cases']);check_inputs(reg)
        write_json_atomic(directory/'results.json',result)
        write_json_atomic(directory/'completion.json',{'cases':15,'registration_sha256':digest(directory/'registration.json'),
            'results_sha256':digest(directory/'results.json'),'completed_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
        print(json.dumps({'completed':directory.relative_to(BASE).as_posix(),**result['selection_gate']}),flush=True)
    except Exception as exc:
        write_json_atomic(directory/'failure.json',{'error':str(exc),'completed_cases':list(result['cases'])});raise


def verify(directory):
    reg=read(directory/'registration.json');result=read(directory/'results.json');completion=read(directory/'completion.json')
    require(len(result['cases'])==completion['cases']==15 and digest(directory/'registration.json')==completion['registration_sha256'] and
            digest(directory/'results.json')==completion['results_sha256'],'归档不完整或已改变')
    require(digest(directory/'registered_plan.md')==reg['registered_plan_sha256'],'登记计划副本变化')
    for name in CODE: require(digest(directory/'registered_code'/name)==digest(Path(__file__).with_name(name)),'登记源码副本变化')
    check_inputs(reg);frames=PrimaryMarketDays(corrected_market());prices,_,bank,proof=feature_bank(frames)
    require(proof==result['feature_verification'],'信号特征核对不同')
    benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True);prior=read(PRIOR/'results.json')['cases']
    hashes={};days=signals=0;maximum=0.;risk_exceedances=0
    for key,item in result['cases'].items():
        path=directory/item['artifact'];require(digest(path)==item['sha256'],'回放JSON已改变');data=read(path)
        checked=inspect_case(frames,reg,item['scenario'],reg['actions'],data)
        require(checked==item['ledger'],'独立账本与运行核查不同')
        signal_proof=verify_signals(frames,prices,bank,reg,item['variant'],data)
        require(signal_proof==item['signals'],'独立目标核对不同')
        computed=metrics_for(data,benchmark);require(all(computed[k]==item[k] for k in computed),'落盘指标不同')
        require(paired_metrics(item,prior[key])==item['paired_comparison'],'配对比较不同')
        csv=directory/(key+'_report.csv');pd.testing.assert_frame_equal(pd.read_csv(csv),pd.DataFrame(data['report']),check_dtype=False,atol=1e-7,rtol=1e-12)
        days+=checked['days'];signals+=signal_proof['signals_verified'];risk_exceedances+=signal_proof['post_budget_target_volatility_above_15pct_days']
        maximum=max(maximum,checked['maximum_absolute_numeric_error'])
        for p in (path,csv): hashes[p.relative_to(BASE).as_posix()]=digest(p)
    require(decision_summary(result['cases'])==result['selection_gate'],'准入研究门槛不同');check_inputs(reg)
    for name in ('registration.json','registered_plan.md','results.json','completion.json'):
        p=directory/name;hashes[p.relative_to(BASE).as_posix()]=digest(p)
    output={'passed':True,'cases_verified':15,'ledger_rows':days,'signals_verified':signals,
        'independent_feature_scores':proof['valid_symbol_scores'],'maximum_ledger_error':maximum,
        'post_budget_target_risk_exceedance_case_days':risk_exceedances,'artifact_hashes':hashes,
        'all_three_stress_complete_for_all_variants':True,**result['selection_gate']}
    write_json_atomic(directory/'verification.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='artifact_hashes'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['register','run','verify']);parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.mode=='register': register()
    else:
        directory=(args.directory or BASE/read(OUT/'latest_registration.json')['directory']).resolve()
        require(directory.is_relative_to(OUT),'需要本研究登记目录')
        run(directory) if args.mode=='run' else verify(directory)
