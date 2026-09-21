"""固定分组、趋势代表与风险贡献均衡；只消费当日及以前的修正总收益信号。"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from research.ex_nasdaq.core import SYMBOLS
from research.goal30.turnover_budget import require
from src.portfolio import risk_group


GROUPS = {
    'large_broad': ['SH510300', 'SH510050', 'SZ159901'],
    'small_broad': ['SH510500', 'SH512100'],
    'growth_broad': ['SZ159915', 'SH588000'],
    'dividend_style': ['SH510880'],
    'financial': ['SH512880', 'SH512800'],
    'healthcare': ['SH512010', 'SH512170'],
    'consumption': ['SZ159928', 'SH512690', 'SZ159996'],
    'technology': ['SZ159995', 'SH512720', 'SH515070', 'SH512980', 'SH515880'],
    'industrial_transition': ['SH512660', 'SH515790', 'SH515030', 'SH512580'],
    'resources': ['SH512400', 'SH515220'],
    'property': ['SH512200'], 'hong_kong': ['SZ159920'],
    'gold': ['SH518880'], 'government_bond': ['SH511260']}
GROUP_OF = {s: name for name, items in GROUPS.items() for s in items}
VARIANTS = [
    {'name': 'group4_risk_parity', 'grouped': True, 'slots': 4, 'sizing': 'risk_parity'},
    {'name': 'ungrouped4_equal', 'grouped': False, 'slots': 4, 'sizing': 'equal'},
    {'name': 'group4_equal', 'grouped': True, 'slots': 4, 'sizing': 'equal'},
    {'name': 'group3_risk_parity', 'grouped': True, 'slots': 3, 'sizing': 'risk_parity'},
    {'name': 'group5_risk_parity', 'grouped': True, 'slots': 5, 'sizing': 'risk_parity'}]
RULES = {'lookbacks': [63, 126, 252], 'volatility_days': 63, 'annualization_days': 252,
    'volatility_floor': .05, 'covariance_diagonal_shrinkage': .25,
    'portfolio_volatility_target': .15, 'max_weight': .35, 'max_group_weight': .70,
    'max_gross_weight': .95, 'rebalance_band': .03, 'minimum_positive_horizons': 2}


def validate_rules():
    require(set(GROUP_OF) == SYMBOLS and sum(map(len, GROUPS.values())) == 30, '固定分组未恰好覆盖30只')


def feature_matrix(prices):
    validate_rules()
    require(prices.index.is_unique and prices.index.is_monotonic_increasing and set(prices.columns) == SYMBOLS,
            '信号日期或固定池无效')
    valid = np.isfinite(prices) & prices.gt(0)
    clean = prices.where(valid)
    history = valid.rolling(253, min_periods=253).sum().eq(253)
    returns = clean.pct_change(fill_method=None)
    volatility = (returns.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)).clip(lower=.05)
    momenta = [np.log(clean / clean.shift(n)) * 252 / n for n in RULES['lookbacks']]
    score = sum(momenta) / 3 / volatility
    positive = sum(m.gt(0).astype(int) for m in momenta)
    return score.where(history & positive.ge(2) & score.gt(0)).replace([np.inf, -np.inf], np.nan)


def prefix_scores(prices):
    """独立直接计算每只253个实际价格，不使用滚动特征。"""
    result = {}
    for symbol in prices:
        x = prices[symbol].tail(253).to_numpy(dtype=float)
        if len(x) < 253 or not np.isfinite(x).all() or (x <= 0).any():
            continue
        daily = x[1:] / x[:-1] - 1
        vol = max(float(np.std(daily[-63:], ddof=1)) * math.sqrt(252), .05)
        momentum = [math.log(x[-1] / x[-n-1]) * 252 / n for n in (63, 126, 252)]
        value = sum(momentum) / 3 / vol
        if value > 0 and sum(v > 0 for v in momentum) >= 2:
            result[symbol] = value
    return result


def covariance(prices, symbols):
    values = prices.reindex(columns=symbols).tail(64).to_numpy(dtype=float)
    require(values.shape == (64, len(symbols)) and np.isfinite(values).all() and (values > 0).all(),
            '风险窗口缺少连续真实正价格')
    daily = values[1:] / values[:-1] - 1
    matrix = np.atleast_2d(np.cov(daily, rowvar=False, ddof=1)) * 252
    diagonal = np.diag(np.maximum(np.diag(matrix), 1e-8))
    result = .75 * matrix + .25 * diagonal
    require(np.isfinite(result).all() and np.linalg.eigvalsh(result).min() > 0, '风险矩阵非正定')
    return result


def risk_parity(matrix):
    """循环坐标下降求正风险预算解；失败显式终止，不暗中换分配方法。"""
    n = len(matrix)
    require(matrix.shape == (n, n) and n > 0 and np.isfinite(matrix).all()
            and np.linalg.eigvalsh(matrix).min() > 0, '风险贡献输入无效')
    c = matrix / np.diag(matrix).mean()
    x = 1 / np.sqrt(np.diag(c)) / math.sqrt(n)
    for _ in range(2000):
        for i in range(n):
            b = c[i] @ x - c[i, i] * x[i]
            x[i] = (-b + math.sqrt(b*b + 4*c[i, i]/n)) / (2*c[i, i])
        rc = x * (c @ x)
        if np.max(np.abs(rc - 1/n)) < 1e-11:
            weights = x / x.sum()
            share = weights * (matrix @ weights) / (weights @ matrix @ weights)
            require(np.max(np.abs(share - 1/n)) < 1e-8, '风险贡献没有收敛')
            return weights
    raise ValueError('风险贡献求解未收敛')


def select_symbols(scores, allowed, variant):
    available = {s: float(v) for s, v in scores.items() if s in allowed and np.isfinite(v) and v > 0}
    if variant['grouped']:
        winners = {}
        for items in GROUPS.values():
            group = {s: available[s] for s in items if s in available}
            if not group:
                continue
            best = max(group.values())
            ties = [s for s, v in group.items() if v == best]
            if len(ties) == 1:
                winners[ties[0]] = best
        available = winners
    selected = []
    for value in sorted(set(available.values()), reverse=True):
        ties = sorted(s for s, v in available.items() if v == value)
        if len(ties) > variant['slots'] - len(selected):
            break
        selected.extend(ties)
    return selected


def limit_weights(weights, prices, policy):
    result = {s: min(float(w), policy['max_weight']) for s, w in weights.items() if w > 1e-10}
    for group in {risk_group(s) for s in result}:
        items = [s for s in result if risk_group(s) == group]
        total = sum(result[s] for s in items)
        scale = min(1., policy['max_group_weight'] / total)
        for s in items:
            result[s] *= scale
    if result:
        total = sum(result.values())
        if total > policy['max_gross_weight']:
            result = {s: w * policy['max_gross_weight'] / total for s, w in result.items()}
        symbols = sorted(result)
        c = covariance(prices, symbols)
        w = np.array([result[s] for s in symbols])
        volatility = math.sqrt(float(w @ c @ w))
        scale = min(1., RULES['portfolio_volatility_target'] / max(volatility, 1e-12))
        result = {s: w * scale for s, w in result.items()}
    return result


def decision(previous, scores, prices, eligible, exiting, current, policy, variant):
    """输入价格已截到当日；持久状态仅记录本月选择，不跨月暗自重新训练。"""
    day = prices.index[-1]
    period = str(day.to_period('M'))
    state = copy.deepcopy(previous)
    allowed = set(eligible) - set(exiting)
    valid = {s: float(v) for s, v in scores.items() if s in allowed and np.isfinite(v) and v > 0}
    review = state.get('month') != period
    risk_weights = {}
    if review:
        chosen = select_symbols(valid, allowed, variant)
        if chosen:
            c = covariance(prices, chosen)
            w = risk_parity(c) if variant['sizing'] == 'risk_parity' else np.ones(len(chosen)) / len(chosen)
            risk_weights = dict(zip(chosen, map(float, w)))
            allocation = policy['max_gross_weight'] * len(chosen) / variant['slots']
            state['raw_weights'] = {s: weight * allocation for s, weight in risk_weights.items()}
        else:
            state['raw_weights'] = {}
        state['month'] = period
    # 月内风险退出不可自行重新进入；月初重选才可恢复。
    state['raw_weights'] = {s: w for s, w in state.get('raw_weights', {}).items() if s in valid}
    desired = limit_weights(state['raw_weights'], prices, policy)
    for s in desired:
        if current.get(s, 0) > 0 and abs(desired[s] - current[s]) < policy['rebalance_band']:
            desired[s] = current[s]
    desired = limit_weights(desired, prices, policy)
    sells = sum(max(0., w - desired.get(s, 0)) for s, w in current.items())
    buys = {s: max(0., w - current.get(s, 0)) for s, w in desired.items()}
    scale = min(1., max(0., policy['max_daily_turnover'] - sells) / max(sum(buys.values()), 1e-12))
    targets = {s: min(w, current.get(s, 0)) + buys[s] * scale for s, w in desired.items()}
    targets = {s: w for s, w in targets.items() if w > 1e-10}
    return state, targets, {'monthly_review': review, 'unconstrained_risk_weights': risk_weights}


def execution_policy(old):
    return {**old, 'max_weight': RULES['max_weight'], 'max_group_weight': RULES['max_group_weight'],
        'max_gross_weight': RULES['max_gross_weight'], 'rebalance_band': RULES['rebalance_band'],
        'volatility_policy': 'observe_only',
        'research_allocation': 'group_diversification_v1_with_separate_shrunk_covariance_volatility_limit'}
