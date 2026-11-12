"""红利低波全收益指数参考；不把指数点位当作可成交ETF份额。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.artifacts import write_json_atomic
from src.runs import sha256
from src.valuation.data import build_calendar

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / 'data/formal/index/seed/H20269.csv'
ENDPOINT = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
PREVIEW_REASON = '全收益指数为持有参考，不是可交易ETF账户，不提供调仓指令或账户试算'


def validate_frame(frame, start, end):
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'])
    if frame.date.duplicated().any():
        raise ValueError('红利低波指数日期重复')
    if not frame.index_code.eq('H20269').all():
        raise ValueError('红利低波参考必须使用H20269全收益指数')
    frame = frame.set_index('date').sort_index().loc[start:end]
    calendar = build_calendar(start, end)
    if not frame.index.equals(calendar):
        raise ValueError('红利低波全收益指数缺少交易日，禁止填充或拼接ETF')
    if not np.isfinite(frame.close).all() or (frame.close <= 0).any():
        raise ValueError('红利低波全收益指数点位无效')
    return frame


def prepare_data(directory, start, end, fetch=False, previous_directory=None):
    parent = Path(previous_directory) / 'index_data/H20269.csv' if previous_directory else SEED
    if not parent.exists():
        parent = SEED
    if parent == SEED:
        metadata = json.loads(SEED.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        if metadata['sha256'] != sha256(SEED):
            raise ValueError('红利低波全收益指数种子哈希不匹配')
    else:
        completion = json.loads((Path(previous_directory) / 'completion.json').read_text(encoding='utf-8'))
        if sha256(parent) != completion['artifact_hashes']['index_data/H20269.csv']:
            raise ValueError('上一正式指数快照哈希不匹配')
    frame = pd.read_csv(parent, parse_dates=['date'])
    destination = directory / 'index_data'
    destination.mkdir(exist_ok=False)
    sources = []
    if fetch:
        session = requests.Session()
        session.trust_env = False
        # 按已有冻结历史的最后年份起追加；必须核对重叠，禁止静默改写旧值。
        for year in range(frame.date.max().year, pd.Timestamp(end).year + 1):
            response = session.get(ENDPOINT, params={'indexCode': 'H20269',
                'startDate': f'{year}0101', 'endDate': min(pd.Timestamp(end), pd.Timestamp(year, 12, 31)).strftime('%Y%m%d')},
                timeout=(8, 30))
            response.raise_for_status()
            body = response.json()
            if str(body.get('code')) != '200' or not body.get('data'):
                raise ValueError('中证官网未返回有效全收益指数日线')
            raw = destination / f'H20269_{year}.json'
            raw.write_bytes(response.content)
            incoming = pd.DataFrame(body['data'])
            if not incoming.indexCode.eq('H20269').all() or not incoming.indexNameCnAll.eq('中证红利低波动全收益指数').all():
                raise ValueError('中证返回的指数身份不符合全收益口径')
            incoming['date'] = pd.to_datetime(incoming.tradeDate, format='%Y%m%d')
            calendar = build_calendar(start, end)
            incoming = incoming[incoming.date.isin(calendar)]
            incoming = incoming[['date', 'close']].assign(index_code='H20269')
            old_values = frame.set_index('date').close
            new_values = incoming.set_index('date').close
            overlap = old_values.index.intersection(new_values.index)
            if not np.allclose(old_values.loc[overlap], new_values.loc[overlap], rtol=0, atol=.000001):
                raise ValueError('中证指数历史发生修订，需独立核查，保留原正式快照')
            frame = pd.concat([frame, incoming[incoming.date > frame.date.max()]], ignore_index=True)
            sources.append({'url': response.url, 'sha256': sha256(raw), 'file': raw.name})
    checked = validate_frame(frame, start, end)
    checked.to_csv(destination / 'H20269.csv')
    write_json_atomic(destination / 'manifest.json', {'index_code': 'H20269', 'kind': 'total_return_index',
        'source': 'csindex_official', 'parent_sha256': sha256(parent),
        'data_sha256': sha256(destination / 'H20269.csv'), 'observations': len(checked),
        'start': str(checked.index[0].date()), 'end': str(checked.index[-1].date()), 'sources': sources})
    return checked


def build_strategy(directory, definition, spec, frame, run_id, stats_fn):
    sid = definition['id']
    capital = spec['initial_capital']
    nav = capital * (frame.close / frame.close.iloc[0])
    report = pd.DataFrame({'account': nav, 'cash': 0., 'value': nav})
    report.index.name = 'date'
    metrics, dd = stats_fn(report, capital)
    end = str(frame.index[-1].date())
    context = {'date': end, 'prices': {}, 'eligible': [],
               'capabilities': {'account_preview': False, 'reason': PREVIEW_REASON}}
    payload = {'id': sid, 'name': definition['name'], 'color': definition['color'], 'kind': 'index_benchmark',
        'status': 'formal_reference', 'description': definition['description'], 'version': spec['version'],
        'run_id': run_id, 'data_start': spec['start'], 'data_end': end, 'initial_capital': capital,
        'account_basis': definition['account_basis'], 'metrics': metrics,
        'goals': {'cagr': metrics['cagr'] >= spec['goals']['cagr'],
                  'worst_complete_year': bool(metrics['worst_complete_year'] and
                      metrics['worst_complete_year']['return'] >= spec['goals']['worst_complete_year']),
                  'max_drawdown': metrics['max_drawdown'] <= spec['goals']['max_drawdown']},
        'investment_admission': False, 'reference_only': True,
        'benchmark': {'index_code': 'H20269', 'price_index_code': 'H30269',
                      'close': float(frame.close.iloc[-1]), 'dividend_reinvestment': True,
                      'exposure': 1., 'source': '中证指数官网'},
        'limitations': ['指数参考，非可交易ETF账户', '含股息再投，未扣基金费与复制成本',
                        '稳健/平衡/积极是展示分类，不表示收益或亏损保证'],
        'capabilities': context['capabilities'], 'cash': 0., 'dividend_receivable': 0., 'holdings': [],
        'advice': {'signal_date': end, 'execution_date': None, 'account': float(nav.iloc[-1]),
                   'cash': 0., 'rows': [], 'orders_count': 0, 'reference_only': True,
                   'basis': PREVIEW_REASON, 'decision_detail': '100%全收益指数持有参考'},
        'series': {'dates': frame.index.strftime('%Y-%m-%d').tolist(), 'capital': nav.tolist(),
                   'drawdown': (-dd).tolist(), 'cash': [0.] * len(nav)},
        'weekly_holdings': [],
        'validation': {'weekly_t_plus_1': None, 'trade_count': 0, 'issues': [],
                       'complete_daily_index': True, 'ledger': None}}
    write_json_atomic(directory / f'{sid}.json', payload)
    write_json_atomic(directory / f'{sid}_context.json', context)
    report.to_csv(directory / f'{sid}_report.csv')
    pd.DataFrame(columns=['date', 'signal_date', 'symbol', 'side', 'quantity', 'price', 'notional']).to_csv(
        directory / f'{sid}_trades.csv', index=False)
    return payload
