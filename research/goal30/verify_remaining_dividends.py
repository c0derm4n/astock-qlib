"""以独立pdfplumber提取、交易所索引及完整页观测核验33项已知分红。"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.acquire_remaining_dividends import ROOT
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.remaining_dividend_evidence import extract_sources
from research.goal30.turnover_budget import require


def independent_fields(path):
    """不调用主提取器；独立PDF引擎按相邻标签读字段，并另查领取句。"""
    sys.path.append('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        raw='\n'.join(page.extract_text() or '' for page in pdf.pages)
        amount_rows=[row for page in pdf.pages for table in page.extract_tables() for row in table
                     if any('本次分红方案' in (cell or '') for cell in row)]
    text=re.sub(r'\s+','',raw).replace('（','(').replace('）',')').replace('：',':')
    def one(pattern):
        hits=re.findall(pattern,text)
        require(hits and len(set(hits))==1,'独立提取缺失或冲突：'+pattern)
        return hits[0]
    def dt(label):
        y,m,d=one(label+r':?(\d{4})年(\d{1,2})月(\d{1,2})日')
        return date(int(y),int(m),int(d)).isoformat()
    # 按表格边界读取，避免换行的单位与右列金额交错而拼成“元/100.53”。
    require(len(amount_rows)==1,'独立提取金额表格不唯一')
    cells=[re.sub(r'\s+','',c or '').replace('（','(').replace('）',')').replace('：',':') for c in amount_rows[0]]
    require(re.fullmatch(r'本次分红方案\(单位:(?:人民币)?元/10份基金份额\)',cells[0]),'独立金额单位不同')
    amounts=[c for c in cells[1:] if c]
    require(len(amounts)==1 and re.fullmatch(r'\d+(?:\.\d+)?',amounts[0]),'独立金额单元格无效')
    amount=Decimal(amounts[0])/10
    fields={'cash_per_share_decimal':str(amount),'record_date':dt('权益登记日'),
            'ex_date':dt('除息日'),'announced_payment_date':dt('现金红利发放日')}
    claims=re.findall(r'投资者可在([^,，。；;]+?)领取现金红利',text)
    require(claims,'独立读取未找到投资者领取句')
    conflicts=[]
    for claim in claims:
        if claim=='办理指定交易后的第二个交易日':
            continue  # 未指定交易的领取条件另列，不能替换普通发放日。
        if claim not in ('现金红利发放日','红利发放日'):
            hit=re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日',claim)
            require(hit is not None,'独立提取遇到未支持领取语义')
            value=date(*(int(x) for x in hit.groups())).isoformat()
            if value!=fields['announced_payment_date']: conflicts.append(value)
    return {'fields':fields,'symbol':'SH'+one(r'基金主代码([0-9]{6})(?![0-9])'),
            'announcement_date':dt('公告送出日期'),'conflicts':conflicts}


def verify():
    require(not (ROOT/'verified_remaining_dividends.json').exists(),'不得覆盖已验收证据')
    scope=read(ROOT/'scope.json');old=read(ROOT/'sources.json');union=read(ROOT/'source_union.json')
    inventory=read(ROOT/'prepared_inventory.json');supp=read(ROOT/'supplement_sources.json')
    complete=read(ROOT/'acquisition_completion.json');visual=read(ROOT/'visual_review.json')
    inputs=dict(scope['input_hashes'])
    for rel,h in inputs.items(): require(digest(BASE/rel)==h,'冻结输入已改变：'+rel)
    require(complete['sources_sha256']==digest(ROOT/'sources.json') and
            complete['scope_sha256']==digest(ROOT/'scope.json'),'首次采集记录不符')
    require(complete['windows']==33 and complete['pdfs']==24 and len(complete['zero_candidate_windows'])==9,
            '首次采集事实不同')
    require(supp['original_sources_sha256']==digest(ROOT/'sources.json'),'补充来源链失效')
    for rel,h in union['input_hashes'].items(): require(digest(ROOT/rel)==h,'原件并集输入失效')
    require(inventory['source_union_sha256']==digest(ROOT/'source_union.json'),'原件目录链失效')
    for item in [supp,inventory]:
        require(item['preparation_code_sha256']==digest(Path(__file__).with_name('prepare_remaining_dividends.py')),
                '准备源码改变')
    expected={e['symbol']+'_'+e['ex_date']:e for e in scope['events']}
    require(len(expected)==33 and len(old['indexes'])==33 and old['complete'],'事件窗口不完整')
    index_rows={};index_pages=0
    for window in old['indexes']:
        event=window['event'];key=event['symbol']+'_'+event['ex_date'];day=date.fromisoformat(event['ex_date'])
        require(event==expected[key] and key not in index_rows,'窗口事件缺失或重复')
        require(len(window['files'])==window['page_count'],'窗口分页缺失')
        rows=[]
        for number,file in enumerate(window['files'],1):
            path=ROOT/file;meta=read(path.with_suffix('.metadata.json'));data=read(path);params=meta['parameters']
            require(meta['sha256']==digest(path) and meta['status']==200 and meta['file']==file,'原始索引损坏')
            require(urlparse(meta['url']).hostname=='query.sse.com.cn','索引来源域名不同')
            require(params['TITLE']=='' and params['SECURITY_CODE']==event['symbol'][2:] and
                    params['START_DATE']==str(day-timedelta(days=35)) and params['END_DATE']==str(day+timedelta(days=7)) and
                    params['pageHelp.pageNo']==number and params['pageHelp.endPage']==number,'索引查询范围不同')
            page=data['pageHelp']
            require(page['pageNo']==number and page['pageCount']==window['page_count'] and
                    page['total']==window['total'] and page['data']==data['result'],'索引分页响应不符')
            require(all(r['SECURITY_CODE']==event['symbol'][2:] and params['START_DATE']<=r['SSEDATE']<=params['END_DATE']
                        for r in data['result']),'索引行越界')
            rows+=data['result'];index_pages+=1
        require(len(rows)==window['total'] and len({r['URL'] for r in rows})==len(rows),'索引条数或重复不符')
        index_rows[key]=rows
    extracted=extract_sources()
    docs={d['file']:d for d in inventory['documents']}
    require(visual['complete'] and len(visual['files'])==33,'完整页目视未完成')
    require(set(visual['files'])==set(docs)=={s['file'] for s in union['sources']},'原件目录与目视集合不同')
    conflicts={};verified=[]
    for item in extracted['events']:
        source=item['source'];file=source['file'];path=ROOT/file;key=source['event_key'];doc=docs[file];view=visual['files'][file]
        matches=[r for r in index_rows[key] if urljoin('https://www.sse.com.cn/',r['URL'])==source['url']]
        require(len(matches)==1 and matches[0]['TITLE']==source['title'] and
                matches[0]['SSEDATE']==source['announcement_date'],'PDF未与原始索引唯一匹配')
        meta=read(path.with_suffix('.metadata.json'))
        require(all(meta[k]==source[k] for k in meta) and meta['status']==200 and
                urlparse(meta['url']).hostname=='www.sse.com.cn' and path.read_bytes().startswith(b'%PDF'), 'PDF下载元数据不符')
        require(digest(path)==doc['pdf_sha256']==view['pdf_sha256']==source['sha256'],'PDF指纹不符')
        require(digest(ROOT/doc['text_file'])==doc['text_sha256'],'文字缓存改变')
        require(len(doc['rendered_hashes'])==doc['pages'],'全页渲染数量不同')
        for rel,h in doc['rendered_hashes'].items(): require(digest(ROOT/rel)==h,'渲染页已改变')
        require(view['pages_reviewed']==item['relevant_pages'],'相关页目视范围不符')
        wanted={p:h for p,h in doc['rendered_hashes'].items() if int(Path(p).stem.rsplit('-',1)[1]) in item['relevant_pages']}
        require(view['rendered_hashes']==wanted,'目视日志包含未核对页或缺页')
        independent=independent_fields(path)
        require(independent['fields']==view['fields_observed']=={k:item['event'][k] for k in independent['fields']},
                '独立PDF引擎、目视及主提取器字段不同')
        require(independent['symbol']==source['symbol'] and independent['announcement_date']==source['announcement_date'],
                '独立身份/公告日期不同')
        require(independent['conflicts']==item['internal_investor_claim_date_conflicts'],'独立领取冲突判定不同')
        require(not item['prior_differences'],'发现与旧日程不同，须单列影响后验收')
        require(item['payment_semantics_verified']==(not independent['conflicts']), '支付语义状态错误')
        if independent['conflicts']: conflicts[key]=independent['conflicts']
        verified.append({**item,'visual_review_completed':True,'primary_fields_verified':True,
                         'independent_extraction':independent})
    require(conflicts=={'SH510880_2017-01-23':['2016-01-26']},'原件冲突集合改变')
    for p in ROOT.rglob('*'):
        if p.is_file(): inputs[p.relative_to(BASE).as_posix()]=digest(p)
    for name in ['remaining_dividend_evidence.py','record_remaining_dividend_review.py',
                 'test_remaining_dividends.py','verify_remaining_dividends.py']:
        p=Path(__file__).with_name(name);inputs[p.relative_to(BASE).as_posix()]=digest(p)
    result={'events':verified,'original_documents':33,'total_pdf_pages':extracted['total_pdf_pages'],
            'visually_reviewed_pages':66,'primary_amount_and_date_table_verified':33,
            'payment_semantics_consistent':32,'internal_payment_date_conflicts':conflicts,
            'visual_review_completed':True,'full_directory_audited':False,
            'actual_broker_receipts_confirmed':False,'investment_admission':False}
    write_json_atomic(ROOT/'verified_remaining_dividends.json',result)
    report={'verified_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'input_hashes':inputs,
            'artifact_hashes':{(ROOT/'verified_remaining_dividends.json').relative_to(BASE).as_posix():digest(ROOT/'verified_remaining_dividends.json')},
            'documents':33,'index_windows':33,'index_pages':index_pages,'pdf_pages':88,'visual_pages':66,
            'independent_pdf_engine':'pdfplumber/pdfminer (main parser uses pypdf)','payment_semantics_consistent':32,
            'internal_payment_date_conflicts':conflicts,'passed':True,'investment_admission':False}
    write_json_atomic(ROOT/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_hashes','artifact_hashes')},ensure_ascii=False))


if __name__=='__main__': verify()
