"""只从发行人原件的场内份额列提取国债ETF权益与投资者到账日期。"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.turnover_budget import require

ROOT = BASE/'output/goal30/data_evidence/broad_readiness_20260914'
DATE = r'(\d{4})年(\d{1,2})月(\d{1,2})日'


def parse_bond_notice(text):
    compact=re.sub(r'\s+','',text).replace('(','（').replace(')','）')
    require('基金主代码511260' in compact and '国泰基金管理有限公司' in compact
            and '上证10年期国债交易型开放式指数证券投资基金' in compact,'公告标的或管理人不符')
    section=compact.split('2与分红相关的其他信息')
    require(len(section)==2,'分红日期表缺失或不唯一')
    table=section[1].split('分红对象')[0]
    labels=[('record_date','权益登记日'),('ex_date','除息日'),('payment_date','现金红利发放日')]
    event={'symbol':'SH511260','type':'cash_dividend','share_class':'exchange_traded'}
    off_exchange,excerpts={},{}
    for i,(field,label) in enumerate(labels):
        stop=labels[i+1][1] if i+1<len(labels) else None
        pieces=table.split(label)
        require(len(pieces)==2,f'{label}不唯一')
        value=pieces[1].split(stop)[0] if stop else pieces[1]
        matches=list(re.finditer(DATE+r'(（场内）|（场外）)?',value))
        require(matches and ''.join(m.group() for m in matches)==value,f'{label}含未识别或缺失内容')
        if any(m[4] for m in matches):
            onsite=[m for m in matches if m[4]=='（场内）']
            offshore=[m for m in matches if m[4]=='（场外）']
            require(len(matches)==2 and len(onsite)==len(offshore)==1,f'{label}场内与场外列不唯一')
            chosen=onsite[0]
            off_exchange[field]=date(*(int(offshore[0][k]) for k in (1,2,3))).isoformat()
        else:
            require(len(matches)==1 and (field=='ex_date' or '本基金本次收益分配仅针对场内份额' in compact),
                    f'{label}未说明场内份额，不能选默认首日期')
            chosen=matches[0]
        event[field]=date(*(int(chosen[k]) for k in (1,2,3))).isoformat()
        excerpts[field]=value
    amounts=list(re.finditer(r'本次分红方案（单位[：:]元/10份基金份额）(\d+(?:\.\d+)?)',compact))
    require(len(amounts)==1,'每10份分红单位或字段无效')
    value=Decimal(amounts[0][1])/10
    require(value>0,'分红必须为正')
    event['cash_per_share']=float(value)
    event['event_id']='SH511260-dividend-'+event['ex_date']
    require(event['record_date']<event['ex_date']<=event['payment_date'],'場内登记、除息及到账日期顺序不符')
    require('上海证券交易所交易结束后' in compact or '下午上海证券交易所交易结束后' in compact,
            '缺少场内收盘登记语义')
    # 原件区分向中登划款和投资者领取；不能把前者当作可用资金日。
    require('投资者可在' in compact and '领取现金红利' in compact,'缺少投资者到账语义')
    return {'event':event,'off_exchange_dates':off_exchange,'table_excerpts':excerpts,
            'amount_excerpt':amounts[0].group()}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    from pypdf import PdfReader
    sources=json.loads((ROOT/'bond_sources.json').read_text(encoding='utf-8'))['sources']
    visual=json.loads((ROOT/'bond_visual_review.json').read_text(encoding='utf-8'))
    index=json.loads((ROOT/'511260_dividend_index.bin').read_bytes())
    meta=json.loads((ROOT/'511260_dividend_index_metadata.json').read_text(encoding='utf-8'))
    require(meta['status']==200 and digest(ROOT/'511260_dividend_index.bin')==meta['sha256'],'公告索引原件改变')
    require(index['pageHelp']['pageCount']==1 and index['pageHelp']['total']==len(index['result'])==len(sources)==4,
            '本次公告查询分页或原件不完整')
    old=json.loads((BASE/'output/goal30/data_evidence/20260913-150124-cc10564d/results.json').read_text(encoding='utf-8'))['symbols']['SH511260']
    em={e['ex_date']:e for e in old['eastmoney_events'] if e['type']=='cash_dividend'}
    sina={e['ex_date']:e for e in old['sina_events'] if e['type']=='cash_dividend'}
    verified=[];inputs={}
    for source in sources:
        p=ROOT/source['file']
        require(digest(p)==source['sha256']==visual['files'][p.name]['pdf_sha256'],'PDF或目视版本不同')
        reader=PdfReader(p)
        require(visual['files'][p.name]['pages_reviewed']==list(range(1,len(reader.pages)+1)),'原件未逐页目视核验')
        for png,expected in visual['files'][p.name]['rendered_hashes'].items():
            require(digest(ROOT/png)==expected,'目视图片指纹改变')
            inputs[(ROOT/png).relative_to(BASE).as_posix()]=expected
        text='\n'.join(page.extract_text() or '' for page in reader.pages)
        parsed=parse_bond_notice(text)
        event=parsed['event'];day=event['ex_date']
        require(source['announcement_date']<event['record_date'] and day in em and day in sina,'公告时点或已知事件集合不同')
        require(abs(event['cash_per_share']-em[day]['cash_per_share'])<1e-12
                and abs(event['cash_per_share']-sina[day]['cash_per_share'])<1e-12,'分红金额与第三方交叉来源不同')
        require(any('https://www.sse.com.cn'+row['URL']==source['url'] and row['SECURITY_CODE']=='511260'
                    and row['SSEDATE']==source['announcement_date'] for row in index['result']),'PDF与官方索引不匹配')
        corrections={k:{'third_party':em[day][k],'exchange_traded':event[k]}
                     for k in ('record_date','payment_date') if em[day][k]!=event[k]}
        verified.append({**parsed,'source':source,'corrections':corrections})
        inputs[p.relative_to(BASE).as_posix()]=digest(p)
    require(len({v['event']['ex_date'] for v in verified})==4,'事件日期重复')
    for p in [ROOT/'bond_sources.json',ROOT/'bond_visual_review.json',ROOT/'511260_dividend_index_metadata.json',
              ROOT/'511260_dividend_index.bin',Path(__file__),BASE/'output/goal30/data_evidence/20260913-150124-cc10564d/results.json']:
        inputs[p.relative_to(BASE).as_posix()]=digest(p)
    return {'symbol':'SH511260','events':sorted(verified,key=lambda v:v['event']['ex_date']),
        'primary_events_verified':4,'corrected_events':sum(bool(v['corrections']) for v in verified),
        'pages_reviewed':sum(len(v['pages_reviewed']) for v in visual['files'].values()),
        'share_class':'exchange_traded','input_hashes':inputs,
        'full_historical_directory_audited':False,'investment_admission':False}


if __name__=='__main__':
    result=verify()
    write_json_atomic(ROOT/'verified_bond_events.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','input_hashes')},ensure_ascii=False))
