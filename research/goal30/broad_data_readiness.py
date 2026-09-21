"""固定30只资产的原始行动资料与场内现金执行字段准备度，非投资准入。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import ROOT, digest
from research.goal30.data_evidence import parse_sina, compare_events
from research.goal30.turnover_budget import require

OLD = BASE/'output/goal30/data_evidence/20260913-150124-cc10564d'
SELECTED = BASE/'output/goal30/data_evidence/selected_candidate'
LISTING = BASE/'output/goal30/verification/listing_dates/verified_listing_dates.json'
REPLAY = BASE/'output/goal30/turnover_replay/20260913-231958-9303749a'
START,END='2016-09-12','2026-09-11'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def action_date(event):
    return event.get('ex_date',event.get('effective_date',event.get('book_date')))


def in_window(events, listing):
    begin=max(START,listing)
    return [e for e in events if begin<=action_date(e)<=END]


def inventory():
    inputs={}
    def checked(path, expected=None):
        value=digest(path)
        require(expected is None or value==expected,f'源文件指纹变化：{path}')
        inputs[Path(path).relative_to(BASE).as_posix()]=value
        return Path(path)
    def loaded(path):
        return read(checked(path))
    old=loaded(OLD/'results.json')['symbols']
    selected=loaded(SELECTED/'results.json')['symbols']
    new=loaded(ROOT/'new_sina_sources.json')['sources']
    listings=loaded(LISTING)['listing_dates']
    bond=loaded(ROOT/'verified_bond_events.json')
    for path,expected in bond['input_hashes'].items():
        checked(BASE/path,expected)
    require(set(listings)==SYMBOLS and len(SYMBOLS)==30,'上市日期不覆盖30只池')
    prior=loaded(REPLAY/'registration.json')
    completion=loaded(REPLAY/'completion.json')
    require(digest(REPLAY/'registration.json')==completion['registration_sha256'],'原现金事件登记改变')
    source=BASE/prior['source']/'market_snapshot.parquet'
    bars=pd.read_parquet(checked(source,prior['input_hashes'][source.relative_to(BASE).as_posix()]))
    calendar=pd.DatetimeIndex(sorted(bars.date.unique()))
    primary={}
    for event in prior['actions']:
        primary.setdefault(event['symbol'],[]).append(event)
    primary['SH511260']=[v['event'] for v in bond['events']]
    rows={}
    for symbol in sorted(SYMBOLS):
        listing=listings[symbol]
        checked(BASE/listing['source_file'],listing['sha256'])
        if symbol in selected:
            value=selected[symbol];directory=SELECTED
            raw=next(s for s in value['sources'] if s.get('file')==symbol+'_sina.txt')
        elif 'comparison' in old[symbol]:
            value=old[symbol];directory=OLD
            raw=next(s for s in value['sources'] if s.get('file')==symbol+'_sina.txt')
        else:
            require(symbol in new and 'source' in new[symbol],f'{symbol}缺少补充原始资料')
            value={};directory=ROOT;raw=new[symbol]['source']
        p=checked(directory/raw['file'],raw['sha256'])
        require(raw['status']==200,'公司行动来源未成功')
        events=parse_sina(p.read_bytes().decode(raw['encoding']),symbol)
        em=value.get('eastmoney_events')
        if em is not None:
            for source_item in value['sources']:
                if source_item.get('file','').endswith('.html'):
                    checked(directory/source_item['file'],source_item['sha256'])
                if source_item.get('archive'):
                    checked(BASE/source_item['archive'],source_item['archive_sha256'])
        start=listing['issuer_list_date']
        sina_events=in_window(events,start)
        em_events=None if em is None else in_window(em,start)
        outside=[{**e,'source':'eastmoney'} for e in em or [] if START<=action_date(e)<start]
        matched=None if em is None else compare_events(em_events,sina_events,start=max(START,start),end=END)
        schedules=[];problems=[]
        for event in sina_events:
            key='cash_per_share' if event['type']=='cash_dividend' else 'ratio'
            primary_match=[p for p in primary.get(symbol,[]) if p['type']==event['type']
                and action_date(p)==action_date(event) and abs(p[key]-event[key])<1e-8]
            if primary_match:
                require(len(primary_match)==1,'原件行动映射不唯一')
                schedules.append({**primary_match[0],'field_evidence':'existing_or_new_primary_notice'})
                continue
            if event['type']=='cash_dividend':
                paired=[v for v in em_events or [] if v['type']==event['type'] and v['ex_date']==event['ex_date']
                        and abs(v[key]-event[key])<=1.1e-5]
                if len(paired)==1 and paired[0].get('payment_date') and paired[0]['record_date']<event['ex_date']<=paired[0]['payment_date']:
                    schedules.append({**paired[0],'field_evidence':'third_party_only_venue_not_primary_verified'})
                else:
                    problems.append({'event':event,'reason':'missing_exchange_traded_record_or_payment_date'})
            else:
                # 新浪生效日与东方财富账面日/比例不同，不静默选一方或四舍五入。
                problems.append({'event':event,'reason':'split_effective_date_ratio_require_primary'})
        if not sina_events and not em_events:
            status='no_actions_observed_not_completeness_proof'
        elif problems:
            status='known_action_fields_incomplete'
        elif all(e['field_evidence']=='existing_or_new_primary_notice' for e in schedules):
            status='known_action_fields_primary_verified'
        else:
            status='known_dividend_schedule_secondary_only'
        # 东方财富有额外的上市后事件也必须保留，不能仅以新浪事件为全集。
        if matched:
            extras=[v for v in matched['unmatched'] if v['reason']!='only_in_sina']
            for row in extras:
                if not any(action_date(row['event'])==action_date(p['event']) and row['event']['type']==p['event']['type'] for p in problems):
                    problems.append({'event':row['event'],'reason':'unmatched_post_listing_eastmoney_event'})
            if problems: status='known_action_fields_incomplete'
        frame=bars[bars.symbol.eq(symbol)].sort_values('date')
        require(len(frame)>0 and frame.date.min()>=pd.Timestamp(start),'上市前行情或缺少行情')
        effective=calendar[(calendar>=max(pd.Timestamp(START),frame.date.min())) & (calendar<=END)]
        missing=effective.difference(pd.DatetimeIndex(frame.date))
        require(np.isfinite(frame[['open','close','adj_close']]).all().all()
                and frame[['open','close','adj_close']].gt(0).all().all(),'行情含无效价格')
        rows[symbol]={'listing_date':start,'first_bar':str(frame.date.min().date()),'last_bar':str(frame.date.max().date()),
            'price_rows':len(frame),'missing_sessions_after_first_bar':[str(d.date()) for d in missing],
            'sina_source':{**raw,'archive':p.relative_to(BASE).as_posix()},
            'sina_events_in_tradable_window':sina_events,'eastmoney_events_in_tradable_window':em_events,
            'prelisting_events_excluded':outside,'cross_source_comparison':matched,
            'known_schedules':schedules,'unresolved':problems,'status':status,
            'full_corporate_actions_directory_audited':False}
    checked(Path(__file__))
    counts={status:sum(row['status']==status for row in rows.values()) for status in sorted({r['status'] for r in rows.values()})}
    return {'created_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'start':START,'end':END,'symbols':rows,
        'summary':{'symbols':30,'raw_sina_sources':30,'new_sina_sources':11,
            'prelisting_events_excluded':sum(len(r['prelisting_events_excluded']) for r in rows.values()),
            'primary_bond_dividends':4,'bond_events_corrected':3,'status_counts':counts},
        'input_hashes':inputs,'full_corporate_actions_directory_audited':False,
        'investment_admission':False,'full_goal_achieved':False}


if __name__=='__main__':
    result=inventory()
    write_json_atomic(ROOT/'readiness.json',result)
    rows=[{'symbol':s,**{k:r[k] for k in ('listing_date','first_bar','last_bar','price_rows','status')},
        'sina_events':len(r['sina_events_in_tradable_window']),'unresolved':len(r['unresolved']),
        'prelisting_events_excluded':len(r['prelisting_events_excluded']),
        'missing_sessions':len(r['missing_sessions_after_first_bar'])} for s,r in result['symbols'].items()]
    pd.DataFrame(rows).to_csv(ROOT/'readiness.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(result['summary'],ensure_ascii=False))
