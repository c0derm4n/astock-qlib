"""归档31项剩余场内分红及2项预热分红的交易所事件窗口与候选原件。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.turnover_budget import require

ROOT = BASE/'output/goal30/data_evidence/remaining_dividends_20260914'
REMAINING = BASE/'output/goal30/data_evidence/broad_actions_20260914/remaining_secondary_schedules.json'


def scope():
    events = read(REMAINING)['events']
    for event in events:
        event['scope'] = 'backtest_window'
    for symbol, amount in [('SH510300', .051), ('SH510880', .05)]:
        events.append({'symbol': symbol, 'type': 'cash_dividend', 'ex_date': '2016-01-20',
                       'cash_per_share': amount, 'scope': 'warmup_signal_only'})
    require(len(events) == 33 and len({(e['symbol'],e['ex_date']) for e in events}) == 33, '预期33项事件缺失或重复')
    return sorted(events, key=lambda e: (e['symbol'],e['ex_date']))


def get(session, url, path, *, params=None):
    require(not path.exists(), '不得覆盖原始下载')
    try:
        response = session.get(url, params=params, timeout=(8,25))
        path.write_bytes(response.content)
        metadata = {'url': response.url, 'status': response.status_code, 'file': path.name,
            'retrieved_at': pd.Timestamp.now(tz='UTC').isoformat(), 'sha256': digest(path),
            'content_type': response.headers.get('Content-Type'), 'parameters': params}
        write_json_atomic(path.with_suffix('.metadata.json'), metadata)
        if response.status_code in (429,514):
            raise RuntimeError('交易所限流，停止本批请求，不重试绕过')
        response.raise_for_status()
        return response, metadata
    except Exception as exc:
        write_json_atomic(path.with_suffix('.failure.json'), {'error': str(exc), 'url': url, 'parameters': params,
                          'recorded_at': pd.Timestamp.now(tz='UTC').isoformat()})
        raise


def acquire(root):
    root = Path(root).resolve()
    require(root.is_relative_to(BASE/'output/goal30/data_evidence'), '下载必须置于新证据目录')
    root.mkdir(parents=True, exist_ok=False)
    events = scope()
    previous = BASE/'output/goal30/group_diversification/20260914-074315-640c978e'
    registry = read(previous/'registration.json')
    inputs = dict(registry['input_hashes'])
    inputs.update({p.relative_to(BASE).as_posix():digest(p) for p in [REMAINING, Path(__file__),
        previous/'registration.json', previous/'verification.json', previous/'results.json', previous/'outcome_audit.json']})
    for p, expected in inputs.items():
        require(digest(BASE/p) == expected, f'旧冻结输入改变：{p}')
    write_json_atomic(root/'scope.json', {'events':events, 'input_hashes':inputs,
        'created_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'scope_note':'Known-event windows only. Downloaded candidates are not yet primary-field verified. No broker receipt confirmation or full-directory claim.'})
    sources, indexes = [], []
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.sse.com.cn/'})
        try:
            for event in events:
                key=event['symbol']+'_'+event['ex_date']
                day=pd.Timestamp(event['ex_date'])
                common={'isPagination':'true','pageHelp.pageSize':100,'pageHelp.beginPage':1,'pageHelp.cacheSize':1,
                    'type':'inParams','sqlId':'COMMON_PL_JJXX_JJGG_NEW_L','TITLE':'','SECURITY_CODE':event['symbol'][2:],
                    'BULLETIN_TYPE':'reits01,fund01,reits02,fund02,reits03,fund03,reits04,fund04,reits05,fund05,reits06,fund06',
                    'START_DATE':str((day-pd.Timedelta(days=35)).date()),'END_DATE':str((day+pd.Timedelta(days=7)).date()),
                    'DATE_DESC':1,'DATE_ASC':'','CODE_DESC':'','CODE_ASC':'','OTHER_TYPE':''}
                page, count, rows, page_files = 1, 1, [], []
                while page <= count:
                    params={**common,'pageHelp.pageNo':page,'pageHelp.endPage':page}
                    path=root/f'{key}_index_{page}.bin'
                    response,meta=get(session,'https://query.sse.com.cn/commonQuery.do',path,params=params)
                    data=response.json()
                    count=max(1,int(data['pageHelp']['pageCount']))
                    require(count <= 20 and all(r['SECURITY_CODE']==event['symbol'][2:] for r in data['result']),
                            '查询未按标的过滤或窗口过大')
                    rows.extend(data['result']);page_files.append(path.name);page+=1
                    time.sleep(.2)
                require(len(rows)==int(data['pageHelp']['total']), '分页总条数不符')
                candidates=[r for r in rows if re.search(r'(分红公告|利润分配公告)',r['TITLE'])
                            and r['SSEDATE']<=event.get('record_date',event['ex_date'])]
                indexes.append({'event':event,'files':page_files,'total':len(rows),'page_count':count,
                    'candidate_count':len(candidates),'candidate_urls':[r['URL'] for r in candidates]})
                for i,row in enumerate(candidates,1):
                    url=urljoin('https://www.sse.com.cn/',row['URL'])
                    require(urlparse(url).hostname=='www.sse.com.cn' and urlparse(url).path.lower().endswith('.pdf'),
                            '候选原件不是已知交易所PDF')
                    path=root/f'{key}_notice_{i}.pdf'
                    response,meta=get(session,url,path)
                    require(response.content.startswith(b'%PDF'), '原件响应并非PDF')
                    sources.append({**meta,'event_key':key,'symbol':event['symbol'],'ex_date_scope':event['ex_date'],
                        'title':row['TITLE'],'announcement_date':row['SSEDATE'],'index_files':page_files,
                        'source_kind':'issuer_pdf_on_sse_pending_field_verification'})
                    time.sleep(.2)
                write_json_atomic(root/'sources.json',{'sources':sources,'indexes':indexes,'complete':False})
                print(json.dumps({'event':key,'query_rows':len(rows),'candidates':len(candidates),'windows_done':len(indexes)},ensure_ascii=False),flush=True)
        except Exception as exc:
            write_json_atomic(root/'failure.json',{'error':str(exc),'windows_completed':len(indexes),'pdfs':len(sources)})
            raise
    write_json_atomic(root/'sources.json',{'sources':sources,'indexes':indexes,'complete':True})
    for p,expected in inputs.items():
        require(digest(BASE/p)==expected,f'下载期间旧冻结输入改变：{p}')
    write_json_atomic(root/'acquisition_completion.json',{'windows':len(indexes),'pdfs':len(sources),
        'zero_candidate_windows':[x['event'] for x in indexes if not x['candidate_count']],
        'sources_sha256':digest(root/'sources.json'),'scope_sha256':digest(root/'scope.json'),
        'completed_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'primary_fields_verified':False,
        'investment_admission':False})
    print(json.dumps({'completed':root.relative_to(BASE).as_posix(),'windows':len(indexes),'pdfs':len(sources)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    acquire(parser.parse_args().root)
