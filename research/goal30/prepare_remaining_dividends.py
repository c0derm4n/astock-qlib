"""保留首次索引，补取收益分配标题漏项，并提取/渲染待核验原件。"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

from backend.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT, get
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.turnover_budget import require


def supplement():
    original=read(ROOT/'sources.json')
    require(original['complete'] and not (ROOT/'source_union.json').exists(), '首次采集未完成或补充已经归档')
    sources=[]
    with requests.Session() as session:
        session.trust_env=False
        session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.sse.com.cn/'})
        for item in original['indexes']:
            if item['candidate_count']:
                continue
            event=item['event'];key=event['symbol']+'_'+event['ex_date']
            rows=[r for f in item['files'] for r in read(ROOT/f)['result']
                  if '收益分配公告' in r['TITLE'] and r['SSEDATE']<=event.get('record_date',event['ex_date'])]
            require(len(rows)==1,'收益分配标题不能唯一匹配待查事件')
            row=rows[0]
            response,meta=get(session,urljoin('https://www.sse.com.cn/',row['URL']),ROOT/(key+'_supplement_1.pdf'))
            require(response.content.startswith(b'%PDF'),'补充响应不是PDF')
            sources.append({**meta,'event_key':key,'symbol':event['symbol'],'ex_date_scope':event['ex_date'],
                'title':row['TITLE'],'announcement_date':row['SSEDATE'],'index_files':item['files'],
                'source_kind':'issuer_pdf_on_sse_pending_field_verification',
                'supplement_reason':'Original unfiltered exchange index contained revenue-distribution title omitted by initial dividend-title matcher.'})
            print(json.dumps({'supplemented':key},ensure_ascii=False),flush=True)
            time.sleep(.2)
    write_json_atomic(ROOT/'supplement_sources.json',{'sources':sources,
        'original_sources_sha256':digest(ROOT/'sources.json'),'preparation_code_sha256':digest(Path(__file__))})
    union=original['sources']+sources
    require(len(union)==len({s['event_key'] for s in union})==33,'原件并集未唯一覆盖33个已知事件')
    write_json_atomic(ROOT/'source_union.json',{'sources':sorted(union,key=lambda s:s['event_key']),
        'input_hashes':{p.name:digest(p) for p in (ROOT/'sources.json',ROOT/'supplement_sources.json',ROOT/'scope.json')},
        'primary_fields_verified':False,'full_directory_audited':False})


def extract_render():
    sys.path.append('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    from pypdf import PdfReader
    union=read(ROOT/'source_union.json')
    inventory=[]
    for i,source in enumerate(union['sources'],1):
        path=ROOT/source['file']
        require(digest(path)==source['sha256'],'原件指纹改变')
        reader=PdfReader(path)
        pages=[page.extract_text() or '' for page in reader.pages]
        text_path=path.with_suffix('.txt')
        text_path.write_text('\n\n'.join(f'--- PAGE {n} ---\n'+text for n,text in enumerate(pages,1)),encoding='utf-8')
        prefix=ROOT/'rendered'/path.stem
        prefix.parent.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','125','-png',str(path),str(prefix)],check=True,capture_output=True)
        images=sorted(prefix.parent.glob(prefix.name+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
        require(len(images)==len(pages),'渲染页数与PDF不符')
        inventory.append({'event_key':source['event_key'],'file':source['file'],'pdf_sha256':source['sha256'],
            'pages':len(pages),'text_file':text_path.name,'text_sha256':digest(text_path),
            'rendered_hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in images},
            'page_text_lengths':[len(t) for t in pages]})
        if i%5==0 or i==len(union['sources']):
            print(json.dumps({'pdfs_prepared':i,'pages':sum(x['pages'] for x in inventory)}),flush=True)
    write_json_atomic(ROOT/'prepared_inventory.json',{'documents':inventory,'source_union_sha256':digest(ROOT/'source_union.json'),
        'preparation_code_sha256':digest(Path(__file__)),'visual_review_completed':False})
    print(json.dumps({'prepared':len(inventory),'pages':sum(x['pages'] for x in inventory)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['supplement','extract_render'])
    supplement() if parser.parse_args().mode=='supplement' else extract_render()
