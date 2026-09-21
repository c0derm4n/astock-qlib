"""剩余33项场内分红原件的严格字段提取；目视完成前不升级证据状态。"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from backend.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.turnover_budget import require

DATE = r'(\d{4}年\d{1,2}月\d{1,2}日)'
MANAGERS = {'SH510050':'华夏基金管理有限公司', 'SH510300':'华泰柏瑞基金管理有限公司',
    'SH510880':'华泰柏瑞基金管理有限公司', 'SH510500':'南方基金管理股份有限公司',
    'SH512100':'南方基金管理股份有限公司', 'SH512400':'南方基金管理股份有限公司',
    'SH512720':'国泰基金管理有限公司'}


def compact(text):
    return re.sub(r'\s+','',text).translate(str.maketrans('（）：，','():,'))


def iso(value):
    m=re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日',value)
    require(m is not None,'不支持的日期')
    return date(*(int(v) for v in m.groups())).isoformat()


def unique(matches, label):
    require(matches and len(set(matches))==1, f'{label}缺失或存在冲突')
    return matches[0]


def parse_notice(pages, symbol):
    # 华夏2016的句子跨页，页首独立页码不得混入“办理指定交易”。
    compact_pages=[compact(re.sub(r'^\s*'+str(i)+r'[ \t]*\r?\n','',t,count=1))
                   for i,t in enumerate(pages,1)]
    text=''.join(compact_pages)
    require(symbol in MANAGERS and MANAGERS[symbol] in text,'管理人不符')
    codes=re.findall(r'基金主代码:?([0-9]{6})(?![0-9])',text)
    require(unique(codes,'基金主代码')==symbol[2:],'基金代码不符')
    require('场外' not in text,'存在场外份额，不能用单列模板默认选择场内日期')
    require(any(s in text for s in ('分红公告','利润分配公告','收益分配公告')),'不是收益分配公告')
    amount_pattern=r'本次分红方案\(单位:(?:人民币)?元/10份基金份额\)(\d+(?:\.\d+)?)'
    amount=Decimal(unique(re.findall(amount_pattern,text),'每10份分红'))
    require(amount>0,'分红金额须为正')
    announced=iso(unique(re.findall(r'公告送出日期:'+DATE,text),'公告日期'))
    # 2017红利ETF的除息行第三格为“-”；已目视确认不是另一个日期或份额类别。
    empty_ex_note='-?' if symbol=='SH510880' and announced=='2017-01-17' else ''
    table_pattern=r'权益登记日'+DATE+r'除息日'+DATE+empty_ex_note+r'现金红利发放日'+DATE+r'分红对象'
    table=unique(re.findall(table_pattern,text),'场内登记/除息/支付日期表')
    record,ex,pay=map(iso,table)
    require(announced<=record<ex<=pay,'公告/登记/除息/领取日期顺序不符')
    entitlement='权益登记日下午上海证券交易所交易结束后'
    require(entitlement in text and '登记在册的本基金全体基金' in text,'缺少场内收盘持有权益语义')
    claim_pattern=r'投资者可在((?:现金)?红利发放日|\d{4}年\d{1,2}月\d{1,2}日)领取现金红利'
    claims=re.findall(claim_pattern,text)
    require(claims,'投资者可领取日期缺失')
    conflicts=[iso(c) for c in claims if c not in ('现金红利发放日','红利发放日') and iso(c)!=pay]
    require('未办理指定交易' in text and '保管' in text and
            re.search(r'已(?:在证券公司处)?办理指定交易的投资者',text) and
            re.search(r'投资者可?在办理指定交易(?:并经确认)?后的第二个交易日可?领取现金红利',text),
            '缺少指定交易条件与未指定交易保管说明')
    creation_extra='以及在权益登记日申购且当日未卖出本基金' in text
    needed=[]
    for i,t in enumerate(compact_pages,1):
        if any(x in t for x in ('基金主代码','权益登记日','投资者可在','领取现金红利','未办理指定交易')):
            needed.append(i)
    require(needed,'缺少可目视字段页')
    return {'event':{'symbol':symbol,'type':'cash_dividend','share_class':'exchange_traded_secondary_market',
        'cash_per_share_decimal':str(amount/10),'record_date':record,'ex_date':ex,
        'announced_payment_date':pay,'announcement_date':announced,
        'entitlement_basis':'record_date_close_holdings',
        'payment_semantics':('announced_distribution_date_actual_broker_receipt_unconfirmed' if conflicts
                             else 'scheduled_investor_claim_date_if_designated_broker'),
        'actual_receipt_confirmed':False,'actual_cash_available_date':None},
        'excerpts':{'amount':re.findall(amount_pattern,text),'date_table':list(table),
                    'entitlement':entitlement,'claim_dates':claims,
                    'undesignated_broker':'未办理指定交易'},
        'additional_primary_creation_entitlement_clause':creation_extra,
        'internal_investor_claim_date_conflicts':conflicts,
        'payment_semantics_verified':not bool(conflicts),
        'execution_scope_note':'Secondary-market purchases only. Primary creation and dedicated-seat distribution procedures are not implemented by this account model.',
        'relevant_pages':needed}


def extract_sources():
    sys.path.append('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    from pypdf import PdfReader
    union=read(ROOT/'source_union.json')
    expected={e['symbol']+'_'+e['ex_date']:e for e in read(ROOT/'scope.json')['events']}
    inventory={r['file']:r for r in read(ROOT/'prepared_inventory.json')['documents']}
    extracted=[]
    for source in union['sources']:
        path=ROOT/source['file'];key=source['event_key']
        require(digest(path)==source['sha256']==inventory[path.name]['pdf_sha256'],'原件指纹改变')
        pages=[p.extract_text() or '' for p in PdfReader(path).pages]
        item=parse_notice(pages,source['symbol']);event=item['event'];prior=expected[key]
        require(event['ex_date']==prior['ex_date'] and event['announcement_date']==source['announcement_date'],
                '原件日期与事件/交易所索引不符')
        differences={}
        for field,old_field in [('record_date','record_date'),('ex_date','ex_date'),('announced_payment_date','payment_date')]:
            if old_field in prior and event[field]!=prior[old_field]:
                differences[field]={'prior':prior[old_field],'primary':event[field]}
        if abs(Decimal(event['cash_per_share_decimal'])-Decimal(str(prior['cash_per_share'])))>Decimal('1e-12'):
            differences['cash_per_share']={'prior':prior['cash_per_share'],'primary':event['cash_per_share_decimal']}
        extracted.append({**item,'source':source,'scope':prior['scope'],'prior_differences':differences,
            'payment_semantics_change':prior['scope']=='backtest_window',
            'visual_review_completed':False})
    require(len(extracted)==33 and len({r['source']['event_key'] for r in extracted})==33,'原件提取未覆盖33项')
    return {'events':extracted,'original_documents':33,'total_pdf_pages':sum(v['pages'] for v in inventory.values()),
        'relevant_pages':sum(len(r['relevant_pages']) for r in extracted),'visual_review_completed':False,
        'primary_fields_verified':False,'full_directory_audited':False,'investment_admission':False}


if __name__=='__main__':
    result=extract_sources()
    write_json_atomic(ROOT/'extracted_fields_pending_visual.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False))
    print(json.dumps([{'key':r['source']['event_key'],'pages':r['relevant_pages'],'differences':r['prior_differences']}
                      for r in result['events']],ensure_ascii=False))
