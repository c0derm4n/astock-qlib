"""自融资分数份额回放引擎：周/月首交易日收盘信号，T+1收盘成交。"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ('red', 'bond')
EXPOSURE_CAP = 0.95


def period_mask(index, freq):
    if freq == 'W':
        periods = index.to_period('W-SUN')
    elif freq == 'M':
        periods = index.to_period('M')
    else:
        raise ValueError(f'未知频率: {freq}')
    return np.r_[True, periods[1:] != periods[:-1]]


def rebalance(values, cash, target, rate, min_fee, band):
    """先卖后买、自融资；登记目标先按95%封顶，再按漂移带保留现有权重。

    与共享引擎的差别：封顶只作用于登记目标。成交费用或价格漂移使实际权重略高于95%时
    不再被强行拉回，避免为千分之一级别的偏离每周支付最低佣金。
    """
    values = np.asarray(values, dtype=float).copy()
    target = np.asarray(target, dtype=float).copy()
    total = target.sum()
    if total > EXPOSURE_CAP:
        target *= EXPOSURE_CAP / total
    nav = cash + values.sum()
    current = values / nav if nav > 0 else np.zeros_like(values)
    keep = (current > 0) & (target > 0) & (np.abs(current - target) < band)
    target = np.where(keep, current, target)
    wanted = target * nav
    trades, costs = [], 0.
    for j in np.flatnonzero(wanted < values - 1e-8):
        amount = values[j] - wanted[j]
        fee = max(min_fee, amount * rate)
        values[j] -= amount
        cash += amount - fee
        costs += fee
        trades.append((int(j), 'sell', float(amount), float(fee)))
    needed = np.maximum(wanted - values, 0.)
    count = int(np.count_nonzero(needed > 1e-8))
    available = max(0., cash - min_fee * count)
    if needed.sum() * (1 + rate) > available:
        needed *= available / (needed.sum() * (1 + rate)) if needed.sum() else 0.
    for j in np.flatnonzero(needed > 1e-8):
        amount = needed[j]
        fee = max(min_fee, amount * rate)
        assert cash + 1e-7 >= amount + fee
        values[j] += amount
        cash -= amount + fee
        costs += fee
        trades.append((int(j), 'buy', float(amount), float(fee)))
    assert cash >= -1e-7 and np.all(values >= -1e-7)
    assert abs(values.sum() + cash - (nav - costs)) < 1e-7
    return values, max(0., cash), trades, costs


def replay(prices, targets, start, capital=100000., cost=.001, min_fee=5., delay=1,
           haircut=.006, band=.05, freq='W', no_fill_dates=()):
    path = prices.loc[start:]
    returns = prices.pct_change(fill_method=None).loc[path.index].fillna(0).to_numpy()
    target_values = targets.reindex(path.index)
    assert target_values.notna().all().all(), '目标权重必须覆盖全部回放日'
    mask = period_mask(path.index, freq)
    forbidden = set(pd.to_datetime(list(no_fill_dates)))
    cash, values = float(capital), np.zeros(len(ASSETS))
    queue, records, trades = {}, [], []
    for i, day in enumerate(path.index):
        prior_nav = float(cash + values.sum())
        pnl = 0.
        if i:
            change = values * returns[i]
            pnl = float(change.sum())
            values = values + change
        days = (day - path.index[i - 1]).days if i else 0
        management = float(values[0] * (1 - np.exp(-haircut * days / 365.2425)))
        values[0] -= management
        cost_today = 0.
        if i in queue:
            signal_day, target = queue.pop(i)
            if day in forbidden:
                queue[i + 1] = (signal_day, target)
            else:
                values, cash, fills, cost_today = rebalance(values, cash, target, cost, min_fee, band)
                trades.extend({'date': str(day.date()), 'signal_date': str(signal_day.date()),
                               'asset': ASSETS[j], 'side': side, 'notional': amount, 'cost': fee}
                              for j, side, amount, fee in fills)
        nav = float(cash + values.sum())
        records.append({'date': day, 'nav': nav, 'cash': cash, 'red_value': values[0],
                        'bond_value': values[1], 'pnl': pnl, 'trading_cost': cost_today,
                        'replication_cost': management, 'prior_nav': prior_nav,
                        'signal_today': bool(mask[i])})
        if mask[i] and i + delay < len(path):
            queue[i + delay] = (day, target_values.iloc[i].to_numpy().copy())
    daily = pd.DataFrame(records).set_index('date')
    orders = pd.DataFrame(trades, columns=['date', 'signal_date', 'asset', 'side',
                                           'notional', 'cost'])
    return daily, orders
