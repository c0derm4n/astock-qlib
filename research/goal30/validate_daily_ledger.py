"""预登记后对全部12个归档情景进行逐日JSON恢复与执行一致性核验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import market30
from research.goal30.audit import BASE
from research.goal30.daily_cash_ledger import EventCalendar, fingerprint, freeze_next, new_state, require, settle_day
from research.goal30.daily_simulation import MarketDays

ROOT = BASE/'output/goal30/daily_ledger_validation'
REFERENCE = BASE/'output/goal30/turnover_replay/20260913-231958-9303749a'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    old = read(REFERENCE/'registration.json')
    completed = read(REFERENCE/'completion.json')
    verified = read(REFERENCE/'verification.json')
    require(digest(REFERENCE/'registration.json') == completed['registration_sha256']
            and digest(REFERENCE/'results.json') == completed['results_sha256'] == verified['results_sha256']
            and verified['cases_verified'] == 12, '参考归档未完整验收或已变动')
    inputs = dict(old['input_hashes'])
    files = [REFERENCE/name for name in ('registration.json','completion.json','verification.json','results.json')]
    files += list(REFERENCE.glob('*.csv'))
    files += [Path(__file__).with_name(name) for name in ('daily_cash_ledger.py','daily_simulation.py',
              'test_daily_cash_ledger.py','validate_daily_ledger.py')]
    files += [BASE/'research/goal30/daily.py', BASE/'research/goal30/forward.py', BASE/'output/current_run.json']
    inputs.update({p.relative_to(BASE).as_posix():digest(p) for p in files})
    for path, expected in inputs.items():
        require(digest(BASE/path) == expected, f'冻结输入已变动：{path}')
    directory = ROOT/(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    payload = {'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'reference':REFERENCE.relative_to(BASE).as_posix(), 'input_hashes':inputs,
        'cases':list(read(REFERENCE/'results.json')['cases']), 'expected_cases':12,
        'protocol':{'start':old['protocol']['start'],'end':old['protocol']['end'],
            'state_recovery':'Every trading day: serialize JSON with finite values, restore and verify state hash before continuation.',
            'fixed_signals':'Recompute all three fixed-control signal streams from the daily state and past-price prefix.',
            'dynamic_signals':'Use previously archived targets to isolate execution parity, without claiming independent signal generation.',
            'compare':'Every report, trade, holding and corporate-action field against all twelve frozen archives; max absolute numeric tolerance 1e-7, no relative tolerance.',
            'forward':'Historical validation only; qualified real-forward days remain zero.',
            'promotion':'Do not replace existing daily automation or production pointer.'}}
    write_json_atomic(directory/'registration.json',payload)
    write_json_atomic(ROOT/'latest_registration.json',{'directory':str(directory),'sha256':digest(directory/'registration.json')})
    print(json.dumps({'registered':str(directory),'inputs':len(inputs),'cases':12},ensure_ascii=False),flush=True)


def compare_frame(actual, expected, label):
    require(set(actual.columns) == set(expected.columns) and len(actual) == len(expected), f'{label}字段或行数不同')
    actual=actual.reindex(columns=expected.columns).reset_index(drop=True)
    expected=expected.reset_index(drop=True)
    errors={}
    for column in expected:
        a,b=actual[column],expected[column]
        if pd.api.types.is_numeric_dtype(b):
            require(np.array_equal(a.isna(),b.isna()),f'{label}/{column}缺失值位置不同')
            finite=b.notna()
            require(np.isfinite(a[finite].to_numpy(dtype=float)).all(),f'{label}/{column}数值无效')
            difference=np.abs(a[finite].to_numpy(dtype=float)-b[finite].to_numpy(dtype=float))
            errors[column]=float(difference.max()) if len(difference) else 0.
            require(errors[column] <= 1e-7, f'{label}/{column}差异{errors[column]}')
        else:
            require(a.fillna('').astype(str).equals(b.fillna('').astype(str)),f'{label}/{column}内容不同')
    return errors


def run(directory):
    directory=Path(directory).resolve()
    require(directory.is_relative_to(ROOT),'只能执行已登记的逐日账本验证')
    registered=read(directory/'registration.json')
    for path,expected in registered['input_hashes'].items():
        require(digest(BASE/path)==expected,f'登记输入已变动：{path}')
    with (directory/'started.json').open('x',encoding='utf-8') as stream:
        json.dump({'started_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()},stream)
    old=read(REFERENCE/'registration.json')
    reference=read(REFERENCE/'results.json')
    require(registered['cases']==list(reference['cases']) and len(registered['cases'])==12,'情景范围被更改')
    market=market30(BASE/old['source'])
    frames=MarketDays(market)
    dates=frames.days[(frames.days>=registered['protocol']['start']) & (frames.days<=registered['protocol']['end'])]
    selected={'candidate':old['execution_adapter_candidate'],'policy':old['policy']}
    cases={}
    try:
        for name in registered['cases']:
            began=time.monotonic()
            case=reference['cases'][name]
            scenario=case['scenario']
            catalog=EventCalendar(old['actions'],frames.days,scenario['payment_delay_sessions'])
            signals=pd.read_csv(REFERENCE/f'signals_{name}.csv').set_index('date')
            state=new_state(old['policy']['initial_capital'],frames.context(dates[0])['previous_date'])
            records,trades,actions,holdings,hashes=[],[],[],[],[]
            issues={key:0 for key in case['issues']}
            fixed_checks=0
            for i,day in enumerate(dates):
                context=frames.context(day)
                state,output=settle_day(state,context,old['policy'],catalog,
                                        slippage_multiplier=scenario['slippage_multiplier'])
                output['report']['bench']=float(frames.returns.at[day,'SH510300'])
                records.append(output['report']);trades.extend(output['trades'])
                actions.extend(output['actions']);holdings.extend(output['holdings'])
                for key,value in output['issues'].items():
                    issues[key]+=value
                if i+1<len(dates):
                    targets=json.loads(signals.at[context['date'],'target_weights'])
                    if case['variant']['method']=='fixed':
                        actual=frames.fixed_targets(selected,state,registered['protocol']['start'])
                        require(set(actual)==set(targets) and all(abs(actual[k]-targets[k])<1e-12 for k in targets),
                                f'{name}/{day}每日固定信号与归档不同')
                        targets=actual;fixed_checks+=1
                    state=freeze_next(state,targets,context['close'],frames.adv.loc[day].to_dict(),old['policy'],dates[i+1])
                # 刻意逐日恢复，避免只在同一进程中验证内存状态。
                state=json.loads(json.dumps(state,ensure_ascii=False,allow_nan=False))
                hashes.append({'date':context['date'],'state_sha256':state['state_sha256']})
            require(issues==case['issues'],f'{name}执行问题计数与参考不同')
            errors,artifacts={},{}
            for kind,rows in [('report',records),('trades',trades),('actions',actions),('holdings',holdings)]:
                actual=pd.DataFrame(rows)
                expected=pd.read_csv(REFERENCE/f'{kind}_{name}.csv')
                errors[kind]=compare_frame(actual,expected,f'{name}/{kind}')
                path=directory/f'{kind}_{name}.csv'
                actual.to_csv(path,index=False)
                artifacts[path.name]=digest(path)
            for filename,payload in [(f'state_{name}.json',state)]:
                write_json_atomic(directory/filename,payload)
                artifacts[filename]=digest(directory/filename)
            path=directory/f'state_hashes_{name}.csv'
            pd.DataFrame(hashes).to_csv(path,index=False);artifacts[path.name]=digest(path)
            cases[name]={'days':len(dates),'json_recoveries':len(hashes),'fixed_signals_recomputed':fixed_checks,
                'dynamic_targets_from_archive':case['variant']['method']!='fixed',
                'errors':errors,'issues':issues,'artifact_hashes':artifacts,'seconds':time.monotonic()-began}
            write_json_atomic(directory/'progress.json',{'completed_cases':list(cases),'total_cases':12})
            print(json.dumps({'case':name,'days':len(dates),'fixed_signals':fixed_checks,
                'max_error':max(v for group in errors.values() for v in group.values()),
                'seconds':cases[name]['seconds']},ensure_ascii=False),flush=True)
        for path,expected in registered['input_hashes'].items():
            require(digest(BASE/path)==expected,f'执行期间输入变动：{path}')
        result={'completed_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'cases':cases,
            'inputs_rechecked':len(registered['input_hashes']),'qualified_real_forward_days':0,
            'investment_admission':False,'full_goal_achieved':False}
        write_json_atomic(directory/'results.json',result)
        write_json_atomic(directory/'completion.json',{'cases':12,'results_sha256':digest(directory/'results.json'),
            'registration_sha256':digest(directory/'registration.json')})
        print(json.dumps({'completed':str(directory),'cases':12},ensure_ascii=False),flush=True)
    except Exception as exc:
        write_json_atomic(directory/'failure.json',{'error':str(exc),'completed_cases':list(cases),
            'failed_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register',action='store_true')
    group.add_argument('--run',type=Path)
    args=parser.parse_args()
    register() if args.register else run(args.run)
