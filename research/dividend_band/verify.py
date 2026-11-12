"""独立复核：逐日账本、单遍回撤、逐窗信号、因果前缀与份额重建。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from research.dividend_trend.run import ROOT, digest
from research.valuation_bands.data import build_calendar
from research.dividend_band.engine import replay
from research.dividend_band.signals import indicators, target_weights

ASSETS = ('red', 'bond')


def independent_indicators(panel):
    """逐窗 numpy 实现，与 pandas rolling 独立。"""
    red = panel.red.to_numpy(dtype=float)
    price = panel.price.to_numpy(dtype=float)
    y10 = panel.yield10.to_numpy(dtype=float)
    n = len(price)
    implied = np.full(n, np.nan)
    implied[1:] = red[1:] / red[:-1] * price[:-1] - price[1:]
    dividend = np.full(n, np.nan)
    for i in range(252, n):
        window = implied[i - 251:i + 1]
        if np.isfinite(window).all():
            dividend[i] = window.sum() / price[i] * 100
    erp = dividend - y10
    dividend_pct = np.full(n, np.nan)
    erp_pct = np.full(n, np.nan)
    for i in range(251, n):
        lo = max(0, i - 755)
        for target, series in ((dividend_pct, dividend), (erp_pct, erp)):
            if not np.isfinite(series[i]):
                continue
            window = series[lo:i + 1]
            finite = window[np.isfinite(window)]
            if len(finite) >= 252:
                target[i] = rankdata(finite)[-1] / len(finite) * 100
    ma = np.full(n, np.nan)
    for i in range(249, n):
        ma[i] = price[i - 249:i + 1].mean()
    returns = np.full(n, np.nan)
    returns[1:] = price[1:] / price[:-1] - 1
    vol = np.full(n, np.nan)
    for i in range(59, n):
        window = returns[i - 59:i + 1]
        if np.isfinite(window).all():
            vol[i] = window.std(ddof=1) * np.sqrt(252)
    risk = np.zeros(n, dtype=bool)
    for i in range(59, n):
        risk[i] = bool(np.isfinite(vol[i]) and vol[i] > 0.20)
    for i in range(269, n):
        deep = price[i] < ma[i] * 0.96 and ma[i] < ma[i - 20]
        risk[i] = bool(risk[i] or deep)
    return pd.DataFrame({'dividend_yield': dividend, 'dividend_pct': dividend_pct,
                         'erp': erp, 'erp_pct': erp_pct, 'ma250': ma, 'vol60': vol,
                         'risk': risk}, index=panel.index)


def independent_weight(value, table):
    result = float(table[-1][1])
    for threshold, weight in table:
        if value >= float(threshold):
            result = float(weight)
            break
    return result


def independent_targets(panel, candidate, tables, signals=None):
    signals = independent_indicators(panel) if signals is None else signals
    weight = np.full(len(panel), np.nan)
    if candidate['indicator'] in ('hold', 'bond', 'static'):
        table_based = False
    else:
        table_based = True
        column = signals[candidate['indicator']].to_numpy()
        for i, value in enumerate(column):
            if np.isfinite(value):
                weight[i] = independent_weight(value, tables[candidate['table']])
        last = np.nan
        for i in range(len(weight)):
            if np.isfinite(weight[i]):
                last = weight[i]
            else:
                weight[i] = last
        weight[~np.isfinite(weight)] = 0.
    target = np.zeros((len(panel), 2))
    if table_based:
        if candidate.get('guard'):
            risk = signals.risk.to_numpy()
            weight = np.where(risk, np.minimum(weight, 0.30), weight)
        weight = np.clip(weight, 0., 0.95)
        target[:, 0] = weight
        if candidate.get('idle', 'bond') == 'bond':
            target[:, 1] = np.clip(0.95 - weight, 0., None)
    elif candidate['indicator'] == 'hold':
        target[:, 0] = 0.95
    elif candidate['indicator'] == 'bond':
        target[:, 1] = 0.95
    else:
        target[:, 0] = float(candidate['red'])
        target[:, 1] = 0.95 - float(candidate['red'])
    listed = panel.bond_listed.astype(bool).to_numpy()
    target[~listed, 1] = 0.
    return pd.DataFrame(target, index=panel.index, columns=list(ASSETS))


def independent_units(prices, folder, haircut):
    daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
    trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
    daily.index = pd.DatetimeIndex(daily.index).as_unit('ns')
    observed = prices.loc[daily.index]
    groups = {day: group for day, group in trades.groupby('date')}
    cash, quantity = 100000., np.zeros(2)
    rebuilt = []
    for i, day in enumerate(daily.index):
        p = observed.loc[day].to_numpy(dtype=float)
        if i:
            days = (day - daily.index[i - 1]).days
            quantity[0] *= np.exp(-haircut * days / 365.2425)
        for row in groups.get(day, pd.DataFrame()).itertuples():
            asset = ASSETS.index(row.asset)
            sign = 1 if row.side == 'buy' else -1
            quantity[asset] += sign * row.notional / p[asset]
            cash -= sign * row.notional + row.cost
            assert abs(row.cost - max(5., row.notional * .001)) < 1e-6
        rebuilt.append(cash + float(quantity @ p))
        assert cash >= -1e-6 and quantity.min() >= -1e-6
    return float(np.max(np.abs(np.asarray(rebuilt) - daily.nav.to_numpy())))


def single_pass_metrics(nav):
    peak, largest, trough = float(nav.iloc[0]), 0., nav.index[0]
    for value in nav:
        if value > peak:
            peak = float(value)
        drop = (peak - float(value)) / peak
        if drop > largest:
            largest = float(drop)
            trough = value.name if hasattr(value, 'name') else trough
    year_end = nav.groupby(nav.index.year).last()
    year_return = year_end.pct_change(fill_method=None).dropna()
    year_return = year_return.loc[year_return.index < nav.index[-1].year]
    years = (nav.index[-1] - nav.index[0]).days / 365.2425
    cagr = float(np.expm1(np.log(nav.iloc[-1] / nav.iloc[0]) / years))
    return {'cagr': cagr, 'max_drawdown': largest,
            'worst_full_year': float(year_return.min()),
            'worst_year': int(year_return.idxmin())}


def verify_run(directory, panel, etf_panel, signals, registered, rows):
    maximum_ledger, maximum_metric = 0., 0.
    for row in rows:
        folder = directory / 'cases' / row['id']
        daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
        daily.index = pd.DatetimeIndex(daily.index).as_unit('ns')
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
        error = daily.nav - (daily.prior_nav + daily.pnl - daily.trading_cost
                             - daily.replication_cost)
        maximum_ledger = max(maximum_ledger, float(error.abs().max()))
        assert daily.index.equals(build_calendar(daily.index[0], daily.index[-1]))
        assert np.allclose(daily.cash + daily.red_value + daily.bond_value, daily.nav,
                           rtol=0, atol=1e-7)
        assert daily.cash.ge(-1e-7).all()
        assert daily[['red_value', 'bond_value']].ge(-1e-7).all().all()
        if len(trades):
            assert (trades.signal_date < trades.date).all()
            assert abs(trades.cost.sum() - daily.trading_cost.sum()) < 1e-7
        independent = single_pass_metrics(daily.nav)
        for field in ('cagr', 'max_drawdown', 'worst_full_year'):
            maximum_metric = max(maximum_metric, abs(independent[field] - row[field]))
        assert independent['worst_year'] == row['worst_year']
        assert maximum_metric < 1e-9

    protocol = registered['protocol']
    tables = protocol['band_tables']
    maximum_signal, maximum_target = 0., 0.
    independent = independent_indicators(panel)
    for candidate in protocol['candidates']:
        pandas_side = signals
        for column in ('dividend_yield', 'dividend_pct', 'erp', 'erp_pct', 'ma250', 'vol60'):
            a, b = independent[column].to_numpy(), pandas_side[column].to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            assert (np.isfinite(a) == np.isfinite(b)).all(), (candidate['id'], column)
            maximum_signal = max(maximum_signal, float(np.max(np.abs(a[both] - b[both]))))
        assert (independent.risk.to_numpy() == pandas_side.risk.to_numpy()).all(), candidate['id']
        left = independent_targets(panel, candidate, tables, independent)
        right = target_weights(panel, signals, candidate, tables)
        maximum_target = max(maximum_target, float((left - right).abs().to_numpy().max()))
    assert maximum_signal < 1e-9, maximum_signal
    assert maximum_target < 1e-12, maximum_target

    prefix_errors = {}
    cutoff = '2021-12-31'
    truncated = panel.loc[:cutoff]
    truncated_signals = indicators(truncated)
    for column in ('dividend_yield', 'dividend_pct', 'erp', 'erp_pct', 'ma250', 'vol60'):
        pd.testing.assert_series_equal(truncated_signals[column],
                                       signals[column].loc[truncated.index])
    pd.testing.assert_series_equal(truncated_signals.risk, signals.risk.loc[truncated.index])
    for candidate in protocol['candidates']:
        if candidate['id'] not in protocol['anchors']:
            continue
        prefix_targets = target_weights(truncated, truncated_signals, candidate, tables)
        full_targets = target_weights(panel, signals, candidate, tables).loc[truncated.index]
        pd.testing.assert_frame_equal(prefix_targets, full_targets)
        daily, _ = replay(truncated[['red', 'bond']], prefix_targets, protocol['start'],
                          no_fill_dates=['2018-08-17'])
        stored = pd.read_csv(directory / 'cases' / candidate['id'] / 'daily.csv',
                             parse_dates=['date']).set_index('date').loc[daily.index]
        prefix_errors[candidate['id']] = float((daily.nav - stored.nav).abs().max())
        assert prefix_errors[candidate['id']] < 1e-7

    unit_errors = {}
    for candidate in protocol['candidates']:
        if candidate['id'] not in protocol['anchors']:
            continue
        unit_errors[candidate['id']] = independent_units(
            panel[['red', 'bond']], directory / 'cases' / candidate['id'], .006)
        assert unit_errors[candidate['id']] < 1e-6
    etf_errors = {}
    for candidate in protocol['candidates']:
        if candidate['id'] not in protocol['anchors']:
            continue
        etf_errors[candidate['id']] = independent_units(
            etf_panel[['red', 'bond']], directory / 'cases' / f"{candidate['id']}__etf", 0.)
        assert etf_errors[candidate['id']] < 1e-6

    current = {p: digest(ROOT / p) for p in registered['hashes']}
    assert current == registered['hashes'], '冻结输入或正式状态发生变化'
    assert maximum_ledger < 1e-7
    return {'status': 'passed', 'case_count': len(rows),
            'self_financing_max_error': maximum_ledger,
            'independent_metric_max_error': maximum_metric,
            'independent_signal_max_error': maximum_signal,
            'independent_target_max_error': maximum_target,
            'index_unit_reconstruction_max_error': max(unit_errors.values()),
            'etf_unit_reconstruction_max_error': max(etf_errors.values()),
            'causal_prefix_max_errors': prefix_errors,
            'calendar_complete': True, 'no_same_day_fills': True,
            'protected_inputs_unchanged': True,
            'lot_rounding_and_broker_fills_verified': False,
            'investment_admission': False}
