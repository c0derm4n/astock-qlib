"""归档免费公司行动资料并交叉核对；不把网页资料等同于发行人审计。"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS
from research.goal30.audit import OUT


class SourceRateLimit(RuntimeError):
    pass


def fetch(url, path):
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(url, timeout=(8, 20), headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code in (429, 514):
            raise SourceRateLimit(f'数据源限流，已停止后续请求；Retry-After={response.headers.get("Retry-After", "未提供")}；{url}')
        response.raise_for_status()
    path.write_bytes(response.content)
    encoding = response.apparent_encoding
    return response.content.decode(encoding), {'url': response.url, 'file': path.name,
        'encoding': encoding, 'sha256': hashlib.sha256(response.content).hexdigest(),
        'retrieved_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'status': response.status_code}


def parse_eastmoney(html, code):
    soup = BeautifulSoup(html, 'html.parser')
    if not soup.title or f'({code})' not in soup.title.get_text():
        raise ValueError('基金分红页面代码或标题不符')
    tables = pd.read_html(StringIO(html))
    dividend = [t for t in tables if {'权益登记日', '除息日', '每10份分红', '分红发放日'} <= set(t.columns)]
    split = [t for t in tables if {'拆分折算日', '拆分折算比例'} <= set(t.columns)]
    if len(dividend) != 1 or len(split) != 1:
        raise ValueError('基金分红/折算表缺失或结构不唯一')
    events = []
    for table, marker, kind in [(dividend[0], '暂无分红信息', 'cash_dividend'),
                                 (split[0], '暂无拆分信息', 'split')]:
        if marker in str(table.iloc[0, 0]):
            continue
        for _, row in table.iterrows():
            if kind == 'cash_dividend':
                match = re.fullmatch(r'每10份派现金\s*(\d+(?:\.\d+)?)元', str(row['每10份分红']).strip())
                if not match:
                    raise ValueError(f'无法确认每份分红单位：{row["每10份分红"]}')
                event = {'type': kind, 'record_date': str(pd.Timestamp(row['权益登记日']).date()),
                         'ex_date': str(pd.Timestamp(row['除息日']).date()),
                         'cash_per_share': float(match[1]) / 10.,
                         'payment_date': None if str(row['分红发放日']).strip() in ('---', '--', 'nan')
                                         else str(pd.Timestamp(row['分红发放日']).date())}
                if event['record_date'] > event['ex_date'] or (event['payment_date'] and event['payment_date'] < event['ex_date']):
                    raise ValueError('分红登记、除息或支付日期顺序异常')
            else:
                match = re.fullmatch(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', str(row['拆分折算比例']).strip())
                if not match or float(match[1]) <= 0 or float(match[2]) <= 0:
                    raise ValueError('折算比例含义无法确认')
                event = {'type': kind, 'book_date': str(pd.Timestamp(row['拆分折算日']).date()),
                         'ratio': float(match[2]) / float(match[1])}
            events.append(event)
    return events


def parse_sina(text, symbol):
    prefix = f'var {symbol.lower()}hfq='
    if not text.strip().startswith(prefix):
        raise ValueError('新浪行动资料标的或格式不符')
    data, end = json.JSONDecoder().raw_decode(text.strip()[len(prefix):])
    if not re.fullmatch(r'\s*;?\s*(?:/\*[\s\S]*\*/)?\s*', text.strip()[len(prefix):][end:]):
        raise ValueError('公司行动JSON含未知尾部内容')
    frame = pd.DataFrame(data['data']).rename(columns={'d': 'date', 's': 'shares', 'u': 'cash'})
    frame['date'] = pd.to_datetime(frame['date'])
    frame[['shares', 'cash']] = frame[['shares', 'cash']].astype(float)
    frame = frame.sort_values('date')
    if (frame.date.duplicated().any() or not np.isfinite(frame[['shares', 'cash']]).all().all()
            or frame.shares.le(0).any() or frame.cash.lt(0).any()):
        raise ValueError('公司行动累计状态无效')
    events = []
    for i in range(1, len(frame)):
        before, after = frame.iloc[i-1], frame.iloc[i]
        ratio, dividend = after.shares / before.shares, after.cash - before.cash
        if abs(ratio - 1) > 1e-7:
            events.append({'type': 'split', 'effective_date': str(after.date.date()), 'ratio': float(ratio)})
        if abs(dividend) > 1e-7:
            events.append({'type': 'cash_dividend', 'ex_date': str(after.date.date()), 'cash_per_share': float(dividend)})
    return events


def compare_events(eastmoney, sina, start='2016-09-12', end='2026-09-11'):
    def date(event):
        return event.get('ex_date', event.get('book_date', event.get('effective_date')))
    a = [e for e in eastmoney if start <= date(e) <= end]
    b = [e for e in sina if start <= date(e) <= end]
    matched, differences = [], []
    used = set()
    for event in a:
        field = 'cash_per_share' if event['type'] == 'cash_dividend' else 'ratio'
        candidates = [(i, other) for i, other in enumerate(b) if i not in used and other['type'] == event['type']
                      and abs(event[field] - other[field]) <= 1.1e-5
                      and abs((pd.Timestamp(date(event)) - pd.Timestamp(date(other))).days) <= (0 if field == 'cash_per_share' else 10)]
        if len(candidates) != 1:
            differences.append({'event': event, 'reason': 'missing_or_ambiguous_counterpart'})
            continue
        i, other = candidates[0]
        used.add(i)
        row = {'eastmoney': event, 'sina': other, 'value_difference': event[field] - other[field]}
        if date(event) != date(other):
            row['date_requires_issuer_confirmation'] = True
        if event['type'] == 'cash_dividend' and event['payment_date'] is None:
            row['payment_date_missing'] = True
        matched.append(row)
    differences += [{'event': event, 'reason': 'only_in_sina'} for i, event in enumerate(b) if i not in used]
    return {'window_start': start, 'window_end': end, 'eastmoney_events': len(a), 'sina_events': len(b),
            'matched': matched, 'unmatched': differences,
            'unresolved_date_count': sum(bool(r.get('date_requires_issuer_confirmation')) for r in matched),
            'missing_payment_date_count': sum(bool(r.get('payment_date_missing')) for r in matched),
            'issuer_audit_complete': False}


def collect():
    directory = OUT / 'data_evidence' / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    def one(symbol):
        code = symbol[2:]
        try:
            html, em_source = fetch(f'https://fundf10.eastmoney.com/fhsp_{code}.html', directory / f'{symbol}_eastmoney.html')
            text, sina_source = fetch(f'https://finance.sina.com.cn/realstock/company/{symbol.lower()}/hfq.js',
                                      directory / f'{symbol}_sina.txt')
            em_events, sina_events = parse_eastmoney(html, code), parse_sina(text, symbol)
            return symbol, {'sources': [em_source, sina_source], 'eastmoney_events': em_events,
                            'sina_events': sina_events, 'comparison': compare_events(em_events, sina_events)}
        except Exception as error:
            return symbol, {'error': str(error), 'error_type': type(error).__name__}
    symbols = {}
    rate_limited = False
    for symbol in sorted(SYMBOLS):
        if rate_limited:
            symbols[symbol] = {'error': '前序请求已限流，本标的未请求', 'error_type': 'NotRequestedAfterRateLimit'}
            continue
        if symbols:
            time.sleep(1.)
        _, value = one(symbol)
        symbols[symbol] = value
        rate_limited = value.get('error_type') == 'SourceRateLimit'
    result = {'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'symbols': symbols,
              'scope': '指定30只的2016-09-12至2026-09-11公司行动公开资料交叉核对',
              'historical_universe_audited': False, 'corporate_actions_audited': False,
              'summary': {'symbols_requested': len(SYMBOLS),
                          'sources_fetched_and_parsed': sum('comparison' in v for v in symbols.values()),
                          'unmatched_events': sum(len(v.get('comparison', {}).get('unmatched', [])) for v in symbols.values()),
                          'date_discrepancies': sum(v.get('comparison', {}).get('unresolved_date_count', 0) for v in symbols.values())}}
    write_json_atomic(directory / 'results.json', result)
    write_json_atomic(OUT / 'data_evidence/latest.json', {'directory': str(directory),
                      'results_sha256': hashlib.sha256((directory / 'results.json').read_bytes()).hexdigest()})
    print(json.dumps({'directory': str(directory), **result['summary']}, ensure_ascii=True, indent=2))
    return directory


if __name__ == '__main__':
    collect()
