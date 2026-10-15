"""正式双策略定义：统一周历、固定配置规则与因果年度模型。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import universe
from src.models import sample_frame, ranked_labels, learn_bundle, predict_bundle
from src.temporal import purged_mask
from src.portfolio import constrain_weights

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = frozenset(('SH' if c.startswith(('5', '6')) else 'SZ') + c
                    for c in universe.UNIVERSE if universe.get_asset_class(c) != 'qdii')


def registry():
    return json.loads((ROOT / 'strategy_registry.json').read_text(encoding='utf-8'))


def weekly_sessions(calendar, start):
    dates = calendar[calendar >= pd.Timestamp(start)]
    periods = dates.to_period('W-SUN')
    return dates[np.r_[True, periods[1:] != periods[:-1]]] if len(dates) else dates


def validate_candidate(candidate):
    symbols = set(candidate['weights']) | {s for a in candidate.get('fallbacks', {}).values() for s in a}
    if symbols - SYMBOLS or candidate['family'] != 'fixed':
        raise ValueError('正式固定策略包含禁止标的或不支持的配置')


def bounded_targets(raw, current, returns, policy):
    desired = constrain_weights(raw, returns, policy)
    old = pd.Series(current, dtype=float)
    index = old.index.union(desired.index)
    old, new = old.reindex(index, fill_value=0), desired.reindex(index, fill_value=0)
    small = ((new - old).abs() < policy['rebalance_band']) & old.gt(0) & new.gt(0)
    new.loc[small] = old.loc[small]
    new = constrain_weights(new, returns, policy).reindex(index, fill_value=0)
    diff = new - old
    sells, buys = -diff.clip(upper=0).sum(), diff.clip(lower=0)
    budget = max(0., policy['max_daily_turnover'] - sells)
    if buys.sum() > budget:
        new = old + diff.clip(upper=0) + buys * budget / buys.sum()
    return new[new > 1e-9].to_dict()


def trend_targets(candidate, date, eligible, current, returns, prices, policy, exiting=()):
    history = prices.loc[:date].tail(200)
    gate = history.iloc[-1].gt(history.mean()) & history.notna().sum().eq(200) & history.gt(0).all()
    raw = {}
    for original, weight in candidate['weights'].items():
        alternatives = {original: 1.} if original in eligible else candidate.get('fallbacks', {}).get(original, {})
        for symbol, fraction in alternatives.items():
            if symbol in eligible and symbol not in exiting and bool(gate.get(symbol, False)):
                raw[symbol] = raw.get(symbol, 0.) + weight * fraction
    return bounded_targets(raw, current, returns.loc[:date], policy)


def annual_lightgbm(market, policy):
    """每个自然年只使用年前成熟标签；保留当年同一模型供最新建议复用。"""
    x, y = sample_frame(market)
    ends = market.label_ends.reindex(x.index)
    dates = x.index.get_level_values('datetime')
    parts, folds, latest_bundle = [], [], None
    for year in range(2016, market.calendar[-1].year + 1):
        test = pd.Timestamp(f'{year}-01-01')
        valid = test - pd.DateOffset(years=1)
        train_mask = purged_mask(dates, ends, dates.min(), valid - pd.Timedelta(days=1), valid) & y.notna()
        valid_mask = purged_mask(dates, ends, valid, test - pd.Timedelta(days=1), test) & y.notna()
        test_mask = (dates >= test) & (dates < test + pd.DateOffset(years=1))
        if not test_mask.any():
            continue
        if train_mask.sum() < 200 or valid_mask.sum() < 50:
            folds.append({'year': year, 'available': False})
            continue
        latest_bundle = learn_bundle(x[train_mask], ranked_labels(y[train_mask]), 'lightgbm', policy['seed'],
                                     x[valid_mask], ranked_labels(y[valid_mask]))
        parts.append(predict_bundle(latest_bundle, x[test_mask]))
        folds.append({'year': year, 'available': True, 'train_rows': int(train_mask.sum()),
                      'valid_rows': int(valid_mask.sum()), 'train_last_label_end': str(ends[train_mask].max().date()),
                      'valid_last_label_end': str(ends[valid_mask].max().date()),
                      'best_iteration': latest_bundle['best_iteration']})
        print(f'正式 LightGBM 年度模型 {year} 完成', flush=True)
    if not parts:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=['datetime', 'instrument'])), folds, None
    return pd.concat(parts).sort_index(), folds, latest_bundle
