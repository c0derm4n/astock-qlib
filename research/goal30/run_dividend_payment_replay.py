"""预登记并核对新原件支付语义：93事件窗口和18完整历史情景。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT as EVIDENCE
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import START, END, action_day, corrected_market, read
from research.goal30.remaining_dividend_actions import upgraded_actions, signal_equivalence
from research.goal30.run_group_diversification import (feature_bank, drive as group_drive,
    verify_signals, metrics_for, check_inputs)
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import (PrimaryMarketDays, REFERENCE, SCENARIOS,
    drive as fixed_drive, inspect_case, verify_event_exposure)

OUT=BASE/'output/goal30/dividend_payment_replay'
GROUP=BASE/'output/goal30/group_diversification/20260914-074315-640c978e'
PRIMARY=BASE/'output/goal30/primary_execution/validation/20260914-072954-cc0f11fd'
CODE=('remaining_dividend_actions.py','run_dividend_payment_replay.py')


def register():
    evidence=read(EVIDENCE/'verification.json');group=read(GROUP/'registration.json');fixed=read(PRIMARY/'registration.json')
    actions,changes=upgraded_actions()
    inputs={**group['input_hashes'],**evidence['input_hashes'],**evidence['artifact_hashes']}
    for root in (GROUP,PRIMARY):
        result=read(root/'results.json')
        for item in result['cases'].values():
            p=root/item['artifact'];require(digest(p)==item['sha256'],'旧回放已改变');inputs[p.relative_to(BASE).as_posix()]=digest(p)
        for name in ('registration.json','results.json','completion.json','verification.json'):
            p=root/name;inputs[p.relative_to(BASE).as_posix()]=digest(p)
    for p in [EVIDENCE/'verification.json',*[Path(__file__).with_name(n) for n in CODE]]:
        inputs[p.relative_to(BASE).as_posix()]=digest(p)
    registration={'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'input_hashes':inputs,
        'actions':actions,'changes':changes,'policy':group['policy'],'fixed_policy':fixed['policy'],
        'candidate':fixed['candidate'],'variants':group['protocol']['variants'],'scenarios':SCENARIOS,
        'protocol':{'start':START,'end':END,'event_windows':93,'full_group_cases':15,'full_fixed_controls':3,'total_cases':111,
          'scope':'Evidence upgrade only. Same rules, raw market, signal series, event amounts and ex dates. No strategy tuning.',
          'payment':'Thirty in-window events remove one uncertain-schedule session. 2017 dividend year conflict retains one extra session. Pay after close; future unobserved payments remain receivable.',
          'signal':'Require exact dataframe equality across64238 rows and all69 known events.',
          'verification':'Independent cash/quantity/receivable reconstruction for every case; all31 upgraded in-window events need nonzero entitlement; group decisions rechecked against independent historical score prefixes.',
          'admission':'No live pointer or daily strategy promotion; full directory, actual receipt and126 real-forward sessions remain unresolved.'}}
    check_inputs(registration)
    directory=OUT/(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False);(directory/'registered_code').mkdir()
    for name in CODE: shutil.copyfile(Path(__file__).with_name(name),directory/'registered_code'/name)
    write_json_atomic(directory/'registration.json',registration)
    write_json_atomic(directory/'actions.json',{'events':actions,'changes':changes,'full_directory_audited':False,'actual_broker_receipts_confirmed':False})
    write_json_atomic(OUT/'latest_registration.json',{'directory':directory.relative_to(BASE).as_posix()})
    print(json.dumps({'registered':directory.relative_to(BASE).as_posix(),'cases':111,'inputs':len(inputs)}),flush=True)


def differences(data,prior):
    before=pd.DataFrame(prior['report']).set_index('date');after=pd.DataFrame(data['report']).set_index('date')
    require(before.index.equals(after.index),'对照日历不同')
    # 状态哈希会因来源元数据变化而变，不能用它替代成交/净值影响。
    return {'maximum_account_difference':float((before.account-after.account).abs().max()),
            'terminal_account_difference':float(after.account.iloc[-1]-before.account.iloc[-1]),
            'maximum_cash_difference':float((before.cash-after.cash).abs().max()),
            'changed_nav_days':int(((before.account-after.account).abs()>1e-7).sum()),
            'same_trades':data['trades']==prior['trades']}


def archive_case(directory,key,data,item,result):
    path=directory/(key+'.json');write_json_atomic(path,data)
    pd.DataFrame(data['report']).to_csv(directory/(key+'_report.csv'),index=False)
    result['cases'][key]={**item,'artifact':path.name,'sha256':digest(path)}
    write_json_atomic(directory/'results.json',result)


def run(directory):
    registered=read(directory/'registration.json');check_inputs(registered)
    with (directory/'started.json').open('x',encoding='utf-8') as stream:
        json.dump({'started_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()},stream)
    result={'cases':{},'investment_admission':False,'qualified_real_forward_sessions':0}
    try:
        events=registered['actions'];result['signal_equivalence']=signal_equivalence(events)
        frames=PrimaryMarketDays(corrected_market())
        fixed={'candidate':registered['candidate'],'policy':registered['fixed_policy']}
        fixed_registration={**registered,'policy':registered['fixed_policy']}
        changed={c['event_id'] for c in registered['changes'] if not c['signal_only']}
        for ordinal,event in enumerate([e for e in events if e['event_id'] in changed],1):
            for scenario in registered['scenarios']:
                start_i=max(1,frames.days.searchsorted(pd.Timestamp(event['record_date']))-8)
                end_i=min(len(frames.days)-1,frames.days.searchsorted(pd.Timestamp(event['announced_payment_date']))+9+scenario['payment_delay_sessions'])
                data=fixed_drive(frames,fixed,[event],scenario,frames.days[start_i:end_i+1],event['symbol'],action_day(event))
                checked=inspect_case(frames,fixed_registration,scenario,[event],data)
                exposure=verify_event_exposure(event,checked,data,scenario,frames.days)
                archive_case(directory,f'event_{ordinal:02d}__'+scenario['name'],data,
                    {'kind':'event','event_id':event['event_id'],'scenario':scenario,'ledger':checked,'exposure':exposure},result)
            if ordinal%10==0 or ordinal==31: print(json.dumps({'event_cases_complete':ordinal*3,'total':93}),flush=True)
        benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True)
        dates=frames.days[(frames.days>=START)&(frames.days<=END)]
        for scenario in registered['scenarios']:
            key='fixed_control__'+scenario['name'];began=time.monotonic()
            data=fixed_drive(frames,fixed,events,scenario,dates)
            item={'kind':'fixed','scenario':scenario,'ledger':inspect_case(frames,fixed_registration,scenario,events,data),
                  **metrics_for(data,benchmark),'prior_difference':differences(data,read(PRIMARY/(key+'.json')))}
            archive_case(directory,key,data,item,result)
            print(json.dumps({'case':key,'cagr':item['metrics']['strategy']['cagr'],'difference':item['prior_difference'],
                              'seconds':time.monotonic()-began}),flush=True)
        prices,features,bank,proof=feature_bank(frames);result['feature_verification']=proof
        for variant in registered['variants']:
            for scenario in registered['scenarios']:
                key=variant['name']+'__'+scenario['name'];began=time.monotonic()
                data=group_drive(frames,prices,features,registered,variant,scenario)
                item={'kind':'group','variant':variant,'scenario':scenario,
                      'ledger':inspect_case(frames,registered,scenario,events,data),
                      'signals':verify_signals(frames,prices,bank,registered,variant,data),
                      **metrics_for(data,benchmark),'prior_difference':differences(data,read(GROUP/(key+'.json')))}
                archive_case(directory,key,data,item,result)
                print(json.dumps({'case':key,'cagr':item['metrics']['strategy']['cagr'],'difference':item['prior_difference'],
                                  'seconds':time.monotonic()-began}),flush=True)
        require(len(result['cases'])==111,'预登记回放未全部完成');check_inputs(registered)
        write_json_atomic(directory/'completion.json',{'cases':111,'registration_sha256':digest(directory/'registration.json'),
             'results_sha256':digest(directory/'results.json'),'completed_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
    except Exception as exc:
        write_json_atomic(directory/'failure.json',{'error':str(exc),'completed_cases':list(result['cases'])});raise


def verify(directory):
    reg=read(directory/'registration.json');result=read(directory/'results.json');completion=read(directory/'completion.json')
    check_inputs(reg)
    require(len(result['cases'])==completion['cases']==111 and digest(directory/'registration.json')==completion['registration_sha256'] and
            digest(directory/'results.json')==completion['results_sha256'],'未完整完成或归档改变')
    require(signal_equivalence(reg['actions'])==result['signal_equivalence'],'信号版本相等性改变')
    frames=PrimaryMarketDays(corrected_market());prices,features,bank,proof=feature_bank(frames)
    require(proof==result['feature_verification'],'价格前缀核验不同')
    benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True)
    events={e['event_id']:e for e in reg['actions']};rights=set();paid=set();pending=set();days=0;hashes={}
    for key,item in result['cases'].items():
        path=directory/item['artifact'];require(digest(path)==item['sha256'],'逐日归档改变');data=read(path)
        policy=reg if item['kind']=='group' else {**reg,'policy':reg['fixed_policy']}
        scope=[events[item['event_id']]] if item['kind']=='event' else reg['actions']
        checked=inspect_case(frames,policy,item['scenario'],scope,data)
        require(checked==item['ledger'],'独立重建与回放不同');days+=checked['days']
        csv=pd.read_csv(directory/(key+'_report.csv'));frame=pd.DataFrame(data['report'])
        pd.testing.assert_frame_equal(csv,frame,check_dtype=False,atol=1e-7,rtol=1e-12)
        if item['kind']=='event':
            require(verify_event_exposure(scope[0],checked,data,item['scenario'],frames.days)==item['exposure'],'事件暴露不同')
            rights.update(checked['nonzero_dividend_rights'])
            paid.update(k for k,v in checked['payments'].items() if v>0)
            if 'beyond_observation' in item['exposure']: pending.add(item['event_id'])
        else:
            metrics=metrics_for(data,benchmark)
            require(all(metrics[k]==item[k] for k in metrics),'落盘指标重算不同')
            root=GROUP if item['kind']=='group' else PRIMARY
            require(differences(data,read(root/(key+'.json')))==item['prior_difference'],'与旧版对照不同')
            if item['kind']=='group':
                require(verify_signals(frames,prices,bank,reg,item['variant'],data)==item['signals'],'决策前缀重算不同')
        hashes[path.relative_to(BASE).as_posix()]=digest(path)
        p=directory/(key+'_report.csv');hashes[p.relative_to(BASE).as_posix()]=digest(p)
    require(len(rights)==31 and len(paid)==30 and pending=={'SH512400:cash_dividend:2026-09-09'},'实际暴露或未来应收覆盖不同')
    check_inputs(reg)
    output={'passed':True,'cases_verified':111,'ledger_days':days,'nonzero_dividend_events':len(rights),
        'nonzero_model_payments':len(paid),'payments_beyond_observation':sorted(pending),
        'complete_history_cases':18,'complete_history_performance_passes':sum(i['metrics']['checks']['joint_admission_performance']
            for i in result['cases'].values() if i['kind']!='event'),
        'artifact_hashes':hashes,'investment_admission':False,'qualified_real_forward_sessions':0}
    for name in ('actions.json','registration.json','results.json','completion.json'):
        p=directory/name;output['artifact_hashes'][p.relative_to(BASE).as_posix()]=digest(p)
    write_json_atomic(directory/'verification.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='artifact_hashes'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['register','run','verify'])
    parser.add_argument('--directory',type=Path);args=parser.parse_args()
    if args.mode=='register': register()
    else:
        directory=(args.directory or BASE/read(OUT/'latest_registration.json')['directory']).resolve()
        require(directory.is_relative_to(OUT),'必须使用本轮登记目录')
        run(directory) if args.mode=='run' else verify(directory)
