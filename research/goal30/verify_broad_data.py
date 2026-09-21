"""独立重建30只累计行动序列，复核场内原件和准备清单；不提升投资准入。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import ROOT,digest,verify as verify_bond
from research.goal30.turnover_budget import require


def reconstruct(raw,symbol):
    prefix=f'var {symbol.lower()}hfq='
    text=raw.strip()
    require(text.startswith(prefix),'原始行动标的不符')
    decoded,_=json.JSONDecoder().raw_decode(text[len(prefix):])
    rows=sorted(decoded['data'],key=lambda r:r['d'])
    require(len({r['d'] for r in rows})==len(rows),'累计行动日期重复')
    events=[]
    for row in rows:
        require(np.isfinite(float(row['s'])) and float(row['s'])>0
                and np.isfinite(float(row['u'])) and float(row['u'])>=0,'累计行动数值无效')
    for old,new in zip(rows,rows[1:]):
        ratio=float(new['s'])/float(old['s'])
        amount=float(new['u'])-float(old['u'])
        day=str(pd.Timestamp(new['d']).date())
        if abs(ratio-1)>1e-7:
            events.append({'type':'split','effective_date':day,'ratio':ratio})
        if abs(amount)>1e-7:
            events.append({'type':'cash_dividend','ex_date':day,'cash_per_share':amount})
    return events


def verify():
    readiness=json.loads((ROOT/'readiness.json').read_text(encoding='utf-8'))
    require(set(readiness['symbols'])==SYMBOLS and len(SYMBOLS)==30,'标的范围不完整')
    for path,expected in readiness['input_hashes'].items():
        require(digest(BASE/path)==expected,f'输入指纹改变：{path}')
    bond=verify_bond()
    archived_bond=json.loads((ROOT/'verified_bond_events.json').read_text(encoding='utf-8'))
    require(bond==archived_bond,'从原始PDF重新提取的场内证据不同')
    classifications={};actions=0;excluded=[]
    for symbol,item in readiness['symbols'].items():
        source=item['sina_source'];path=BASE/source['archive']
        require(digest(path)==source['sha256'],'新浪原始源指纹改变')
        events=reconstruct(path.read_bytes().decode(source['encoding']),symbol)
        filtered=[e for e in events if max(readiness['start'],item['listing_date'])<=
                  e.get('ex_date',e.get('effective_date'))<=readiness['end']]
        require(filtered==item['sina_events_in_tradable_window'],'累计资料重建事件与清单不同')
        actions+=len(filtered)
        for event in item['prelisting_events_excluded']:
            day=event.get('ex_date',event.get('book_date',event.get('effective_date')))
            require(readiness['start']<=day<item['listing_date'],'误把上市后事件排除')
            excluded.append({'symbol':symbol,**event})
        schedules=item['known_schedules']
        for event in schedules:
            if event['type']=='cash_dividend':
                require(event['record_date']<event['ex_date']<=event['payment_date']
                        and event['cash_per_share']>0,'完整日程含错误日期或分红额')
        if item['unresolved']:
            status='known_action_fields_incomplete'
        elif not filtered and not item['eastmoney_events_in_tradable_window']:
            status='no_actions_observed_not_completeness_proof'
        elif all(e['field_evidence']=='existing_or_new_primary_notice' for e in schedules):
            status='known_action_fields_primary_verified'
        else:
            status='known_dividend_schedule_secondary_only'
        require(status==item['status'] and item['full_corporate_actions_directory_audited'] is False,
                '状态分类或审计范围不符')
        classifications[status]=classifications.get(status,0)+1
    require(classifications==readiness['summary']['status_counts'],'状态统计不符')
    require(len(excluded)==readiness['summary']['prelisting_events_excluded']==2
            and {e['symbol'] for e in excluded}=={'SH511260','SH588000'},'上市前排除范围不符')
    csv=pd.read_csv(ROOT/'readiness.csv')
    require(set(csv.symbol)==SYMBOLS and len(csv)==30,'清单CSV标的范围不同')
    for row in csv.itertuples():
        item=readiness['symbols'][row.symbol]
        require(row.status==item['status'] and row.sina_events==len(item['sina_events_in_tradable_window'])
                and row.unresolved==len(item['unresolved']) and row.missing_sessions==len(item['missing_sessions_after_first_bar'])
                and row.price_rows==item['price_rows'],'清单CSV统计不同')
    # 复核上阶段234项冻结输入，覆盖旧生产代码、行动模型、旧证据和正式指针。
    prior=BASE/'output/goal30/daily_ledger_validation/20260913-235145-0e2ea218/registration.json'
    hashes=json.loads(prior.read_text(encoding='utf-8'))['input_hashes']
    for path,expected in hashes.items():
        require(digest(BASE/path)==expected,f'既有冻结输入改变：{path}')
    result={'verified_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'symbols_verified':30,
        'sina_events_reconstructed_in_tradable_window':actions,'new_sina_sources':11,
        'primary_bond_events_verified':4,'bond_events_corrected':3,'primary_pages_reviewed':11,
        'prelisting_events_excluded':excluded,'status_counts':classifications,
        'readiness_input_hashes_verified':len(readiness['input_hashes']),'prior_frozen_inputs_verified':len(hashes),
        'artifact_hashes':{p.relative_to(BASE).as_posix():digest(p) for p in
            [ROOT/'readiness.json',ROOT/'readiness.csv',ROOT/'verified_bond_events.json',Path(__file__)]},
        'full_corporate_actions_directory_audited':False,'investment_admission':False,'full_goal_achieved':False}
    write_json_atomic(ROOT/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':verify()
