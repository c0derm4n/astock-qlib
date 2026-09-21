"""独立检查逐日恢复的归档结果、终态与冻结参考，禁止把模拟记作真实前瞻。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.daily_cash_ledger import fingerprint, require, validate_state


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(directory):
    directory=Path(directory).resolve()
    require(directory.is_relative_to(BASE/'output/goal30/daily_ledger_validation'),'核验目录不符')
    registration=read(directory/'registration.json')
    completion=read(directory/'completion.json')
    result=read(directory/'results.json')
    require(digest(directory/'registration.json')==completion['registration_sha256']
            and digest(directory/'results.json')==completion['results_sha256'],'完成后归档改变')
    for path,expected in registration['input_hashes'].items():
        require(digest(BASE/path)==expected,f'冻结输入改变：{path}')
    source=BASE/registration['reference']
    old=read(source/'results.json')
    require(set(result['cases'])==set(old['cases'])==set(registration['cases'])
            and len(result['cases'])==completion['cases']==12,'12情景范围不完整')
    require(result['qualified_real_forward_days']==0 and result['investment_admission'] is False
            and result['full_goal_achieved'] is False,'历史验证不得取得真实前瞻或投资资格')
    cases={}
    for name,case in result['cases'].items():
        require(len(case['artifact_hashes'])==6,'缺少每日恢复产物')
        for file,expected in case['artifact_hashes'].items():
            require(digest(directory/file)==expected,f'产物指纹不一致：{file}')
        errors={}
        for kind in ('report','trades','actions','holdings'):
            current=pd.read_csv(directory/f'{kind}_{name}.csv')
            reference=pd.read_csv(source/f'{kind}_{name}.csv')
            require(set(current.columns)==set(reference.columns) and len(current)==len(reference),'字段或行数不一致')
            for col in reference:
                a,b=current[col],reference[col]
                require(np.array_equal(a.isna(),b.isna()),f'{name}/{kind}/{col}缺失值改变')
                valid=b.notna()
                if pd.api.types.is_numeric_dtype(b):
                    difference=(a[valid]-b[valid]).abs()
                    error=float(difference.max()) if len(difference) else 0.
                    require(np.isfinite(a[valid]).all() and error<=1e-7,f'{name}/{kind}/{col}数字不同')
                    errors[f'{kind}/{col}']=error
                else:
                    require(a.fillna('').equals(b.fillna('')),f'{name}/{kind}/{col}记录不同')
        daily=pd.read_csv(directory/f'report_{name}.csv',index_col='date')
        inventory=pd.read_csv(directory/f'holdings_{name}.csv')
        state=read(directory/f'state_{name}.json')
        validate_state(state)
        require(state['mode']=='historical_validation' and state['sequence']==case['days']==len(daily)==2428,
                '终态模式或完整交易日数不符')
        require(state['last_date']==daily.index[-1] and state['pending_orders']==[]
                and state['next_execution_date'] is None,'期末日期或未完成信号不符')
        final=daily.iloc[-1]
        require(abs(state['cash']-final['cash'])<1e-6
                and abs(state['previous_nav']-final['account'])<1e-6
                and abs(sum(state['receivable'].values())-final['dividend_receivable'])<1e-6,'终态现金或应收不同')
        holdings=inventory[inventory.date.eq(state['last_date'])].set_index('symbol').quantity.to_dict()
        require(set(holdings)==set(state['quantities'])
                and all(abs(q-state['quantities'][s])<1e-8 for s,q in holdings.items()),'终态份额不同')
        stages=state['processed_stages']
        require(len(stages)==len(set(stages)),'公司行动阶段重复')
        state_hashes=pd.read_csv(directory/f'state_hashes_{name}.csv')
        require(state_hashes.date.tolist()==daily.index.tolist() and case['json_recoveries']==len(daily)
                and state_hashes.state_sha256.iloc[-1]==state['state_sha256'],'恢复记录日期或期末指纹不符')
        require(state_hashes.state_sha256.str.fullmatch('[0-9a-f]{64}').all(),'逐日状态摘要格式无效')
        require((daily.cash>=-1e-6).all() and (daily.dividend_receivable>=0).all()
                and (daily[['turnover','execution_turnover','budget_turnover']]<=.3+1e-10).all().all(),
                '逐日现金或共享预算违反约束')
        fixed=old['cases'][name]['variant']['method']=='fixed'
        require(case['fixed_signals_recomputed']==(2427 if fixed else 0)
                and case['dynamic_targets_from_archive']==(not fixed),'信号验证范围描述错误')
        require(case['issues']==old['cases'][name]['issues'],'订单执行问题计数不同')
        cases[name]={'archive_fields_match':True,'final_state_matches':True,'budget_passed':True,
            'maximum_field_error':max(errors.values()),'days':len(daily),'artifacts_verified':6,
            'fixed_signals_recomputed':case['fixed_signals_recomputed']}
    payload={'verified_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'cases_verified':12,
        'cases':cases,'input_hashes_verified':len(registration['input_hashes']),
        'artifacts_verified':sum(c['artifacts_verified'] for c in cases.values()),
        'daily_state_recoveries':sum(c['days'] for c in cases.values()),
        'fixed_signals_recomputed':sum(c['fixed_signals_recomputed'] for c in cases.values()),
        'results_sha256':digest(directory/'results.json'),'verification_code_sha256':digest(Path(__file__)),
        'qualified_real_forward_days':0,'investment_admission':False,'full_goal_achieved':False}
    write_json_atomic(directory/'verification.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='cases'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
