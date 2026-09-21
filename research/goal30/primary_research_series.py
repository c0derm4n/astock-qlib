"""按已知公司行动重建开发研究信号；原始价格及已登记复权列保持不变。"""
from __future__ import annotations

import copy
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, market30
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.broad_action_evidence import ROOT as ACTIONS_ROOT, READINESS
from research.goal30.data_evidence import parse_sina
from research.goal30.turnover_budget import require

ROOT = BASE / 'output/goal30/primary_execution'
SOURCE = BASE / 'output/runs/20260912-220345-f20ea3f5'
START, END = '2016-09-12', '2026-09-11'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decimal_source(value):
    # 仅去除已存浮点现金值的二进制尾差；精确比例直接沿用原件字符串。
    result = format(float(value), '.12g')
    require(abs(Decimal(str(value)) - Decimal(result)) < Decimal('1e-12'), '浮点规范化超出尾差范围')
    return result


def action_day(action):
    return action['effective_date'] if action['type'] == 'split' else action['ex_date']


def known_actions():
    readiness = read(READINESS)
    new_primary = read(ACTIONS_ROOT / 'primary_events.json')['events']
    legacy = read(BASE / 'output/goal30/verification/legacy_primary/verified_primary.json')
    csi300 = {e['ex_date']: e for e in legacy['known_dividend_events']}
    notices = BASE / 'output/goal30/data_evidence/issuer_notices'
    communications = read(notices / 'verified_events.json')
    for file, expected in communications['source_hashes'].items():
        require(digest(notices / file) == expected, '通信ETF原件指纹改变')
    comm = {e['effective_date']: e for e in communications['split_events']}
    events = []
    for symbol, row in readiness['symbols'].items():
        for old in row['known_schedules']:
            event = {**old, 'symbol': symbol, 'signal_only': False,
                     'evidence_level': 'primary' if old['field_evidence'] == 'existing_or_new_primary_notice' else 'secondary',
                     'evidence_file': READINESS.relative_to(BASE).as_posix()}
            if old['type'] == 'split':
                require(symbol == 'SH515880', '未知旧拆分，需先读取对应原件字段')
                original = comm[old['effective_date']]
                require(original['ratio'] == old['ratio'] and original['record_date'] < old['effective_date'], '通信拆分日程不同')
                event.update(record_date=original['record_date'], book_date=original['record_date'],
                    ratio_decimal=str(original['ratio']), rounding='integer_ratio_no_fraction_for_integer_holdings',
                    evidence_file=(notices / 'verified_events.json').relative_to(BASE).as_posix())
            else:
                cash = old['cash_per_share']
                if symbol == 'SH510300':
                    original = csi300[old['ex_date']]
                    require(all(old[k] == original[k] for k in ('record_date', 'ex_date', 'payment_date')), '沪深300日程与原件不同')
                    cash = original['cash_per_share']
                    event['evidence_file'] = original['file']
                event.update(cash_per_share_decimal=decimal_source(cash),
                    entitlement_basis='record_date_close_holdings', announced_payment_date=old['payment_date'],
                    payment_semantics='legacy_primary_scheduled_distribution' if event['evidence_level'] == 'primary' else 'secondary_schedule_venue_not_primary_verified',
                    actual_receipt_confirmed=False, actual_cash_available_date=None)
                event.pop('payment_date', None)
            events.append(event)
    for item in new_primary:
        event = copy.deepcopy(item['event'])
        event.update(evidence_level='primary', signal_only=False,
                     evidence_file= (ACTIONS_ROOT / 'primary_events.json').relative_to(BASE).as_posix())
        if event['type'] == 'split':
            event['effective_date'] = event['new_units_available_date']
        events.append(event)
    require(len(events) == 67, '交易区间已知事件覆盖数不符')
    # 预热数据从2016-01-04开始，不能漏掉2016-01-20的两项已知现金事件。
    for symbol, row in readiness['symbols'].items():
        source = row['sina_source']
        path = BASE / source['archive']
        require(digest(path) == source['sha256'], '新浪累计行动原件改变')
        for item in parse_sina(path.read_bytes().decode(source['encoding']), symbol):
            if max('2016-01-04', row['listing_date']) <= action_day(item) < START:
                require(item['type'] == 'cash_dividend', '预热期拆分需原件确认')
                events.append({**item, 'symbol': symbol, 'signal_only': True,
                    'cash_per_share_decimal': decimal_source(item['cash_per_share']),
                    'evidence_level': 'secondary', 'evidence_file': source['archive'],
                    'scope_note': 'warmup_signal_amount_only_record_and_payment_not_verified'})
    require(len(events) == 69 and sum(e['signal_only'] for e in events) == 2, '预热期范围不同')
    for event in events:
        event['source_event_id'] = event.get('event_id')
        event['event_id'] = ':'.join((event['symbol'], event['type'], action_day(event)))
        event.pop('field_evidence', None)
    require(len({e['event_id'] for e in events}) == len(events), '合并事件重复')
    return sorted(events, key=lambda e: (action_day(e), e['symbol']))


def rebuild_series(raw, actions):
    """总收益信号采用除息日收盘理论再投资；并非可执行的现金账户。"""
    require(not raw.duplicated(['symbol', 'date']).any(), '行情主键重复')
    require(set(raw.symbol).issubset(SYMBOLS), '行情包含池外标的')
    require(np.isfinite(raw[['open', 'close']]).all().all() and (raw[['open', 'close']] > 0).all().all(), '未复权价格无效')
    rows, applied = [], []
    keys = set()
    for event in actions:
        key = (event['symbol'], action_day(event))
        require(event['symbol'] in SYMBOLS and key not in keys, '同日多项行动需要额外顺序证据，不能猜测')
        keys.add(key)
    for symbol, original in raw.groupby('symbol', sort=True):
        frame = original.sort_values('date').reset_index(drop=True)
        date_to_i = {str(day.date()): i for i, day in enumerate(frame.date)}
        splits = np.ones(len(frame)); cash = np.zeros(len(frame))
        for event in actions:
            if event['symbol'] != symbol:
                continue
            day = action_day(event)
            if day > str(frame.date.iloc[-1].date()):
                continue  # 前缀检验：未来行动不参与历史计算。
            require(day in date_to_i and date_to_i[day] > 0, f'{symbol}/{day}行动缺少前后实际行情')
            i = date_to_i[day]
            if event['type'] == 'split':
                value = Decimal(event['ratio_decimal'])
                require(value.is_finite() and value > 0 and event['book_date'] < day, '拆分比例或价格单位日期无效')
                splits[i] = float(value)
            else:
                require(event['type'] == 'cash_dividend', '未知行动类型')
                value = Decimal(event['cash_per_share_decimal'])
                require(value.is_finite() and value > 0, '现金金额无效')
                cash[i] = float(value)
            applied.append(event['event_id'])
        previous = frame.close.shift(1).to_numpy()
        gross = (frame.close.to_numpy() * splits + cash) / previous
        gross[0] = 1.
        require(np.isfinite(gross).all() and (gross > 0).all(), '总收益变化无效')
        index = 100. * np.cumprod(gross)
        rows.append(pd.DataFrame({'date': frame.date, 'symbol': symbol,
            'signal_total_return_index': index, 'signal_return_since_previous_observed_close': np.where(np.arange(len(frame)) == 0, np.nan, gross - 1),
            'split_ratio': splits, 'cash_dividend_per_share': cash,
            'previous_observed_date': frame.date.shift(1)}))
    result = pd.concat(rows, ignore_index=True).sort_values(['date', 'symbol']).reset_index(drop=True)
    return result, applied


def corrected_market(source=SOURCE, series=None):
    """仅供新研究入口；不复用旧特征/标签，以免混合两个收益口径。"""
    market = market30(source)
    if series is None:
        series = pd.read_parquet(ROOT / 'signal_series.parquet')
    joined = market.bars.merge(series[['date', 'symbol', 'signal_total_return_index']], on=['date', 'symbol'], validate='one_to_one')
    require(len(joined) == len(market.bars) and joined.signal_total_return_index.notna().all(), '信号未覆盖行情')
    joined['original_adj_close'] = joined.adj_close
    joined['adj_close'] = joined.signal_total_return_index
    joined = joined.rename(columns={'adj_factor': 'original_adj_factor', 'adj_open': 'original_adj_open'})
    market.bars = joined
    # 策略只消费收盘总收益与原始成交数据；其他旧特征、标签不得使用。
    market.features = pd.DataFrame()
    market.labels = pd.Series(dtype=float)
    market.label_ends = pd.Series(dtype='datetime64[ns]')
    market.audit = {**market.audit, 'return_basis': 'known_actions_total_return_signal_only',
        'investment_admission': False, 'full_corporate_actions_directory_audited': False}
    return market


def build():
    previous = read(ACTIONS_ROOT / 'verification.json')
    inputs = {**previous['input_hashes'], **previous['artifact_hashes']}
    for path, expected in inputs.items():
        require(digest(BASE / path) == expected, f'既有证据改变：{path}')
    actions = known_actions()
    raw_path = SOURCE / 'market_snapshot.parquet'
    raw = pd.read_parquet(raw_path)
    raw = raw[raw.symbol.isin(SYMBOLS)].copy()
    require(set(raw.symbol) == SYMBOLS, '原始价格未覆盖固定30只')
    series, applied = rebuild_series(raw, actions)
    require(set(applied) == {e['event_id'] for e in actions}, '已知事件未全部参与序列重建')
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ROOT / 'actions.json', {'events': actions,
        'known_events_in_backtest': 67, 'warmup_signal_only_events': 2,
        'evidence_level_counts': dict(Counter(e['evidence_level'] for e in actions)),
        'full_directory_audited': False, 'investment_admission': False})
    series.to_parquet(ROOT / 'signal_series.parquet', index=False)
    for path in (READINESS, ACTIONS_ROOT / 'primary_events.json', raw_path, Path(__file__),
                 BASE / 'output/goal30/data_evidence/issuer_notices/verified_events.json'):
        inputs[path.relative_to(BASE).as_posix()] = digest(path)
    write_json_atomic(ROOT / 'series_manifest.json', {'input_hashes': inputs,
        'artifact_hashes': {p.relative_to(BASE).as_posix(): digest(p) for p in (ROOT / 'actions.json', ROOT / 'signal_series.parquet')},
        'symbols': sorted(SYMBOLS), 'rows': len(series), 'events_applied': applied,
        'model': 'Close total-return signal assumes fractional reinvestment at ex-date close. Execution uses raw prices, integer shares and separate receivables.',
        'start': str(series.date.min().date()), 'end': str(series.date.max().date()),
        'warmup_and_secondary_evidence_remain_provisional': True,
        'investment_admission': False})
    print(json.dumps({'rows': len(series), 'symbols': len(SYMBOLS), 'events': len(applied),
        'primary_events': sum(e['evidence_level'] == 'primary' for e in actions), 'secondary_events': sum(e['evidence_level'] == 'secondary' for e in actions)}, ensure_ascii=False))


if __name__ == '__main__':
    build()
