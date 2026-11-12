"""可复核的指数/ETF价格代理趋势研究；所有输出只新增到独立run。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import uuid

import duckdb
import numpy as np
import pandas as pd
import requests

from research.valuation_bands.data import build_calendar

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('protocol.json')
OLD = ROOT / 'output/valuation_bands/comparison_100k_20261109'
INDEX = ROOT / 'output/valuation_bands/dividend_index_comparison_20261109/official_H20269_daily.csv'
DB = ROOT / 'data/formal/market.duckdb'
ETF = ROOT / 'data/valuation_bands/snapshots/etfrun-equal-20261109-v1/etf_bars.csv'
ENDPOINT = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def todo_write(directory, phase):
    phases = ['data', 'replay', 'validation', 'delivery']
    pos = phases.index(phase) if phase in phases else len(phases)
    write_json(directory / 'status.json', {'phase': phase, 'tasks': [
        {'task': name, 'status': 'completed' if i < pos else
         ('in_progress' if i == pos else 'pending')} for i, name in enumerate(phases)]})


def candidates():
    rows = []
    for ma in (150, 200, 250):
        for vol, suffix in ((.15, 'v15'), (None, 'no_vol')):
            rows.append(dict(id=f'cash95_ma{ma}_{suffix}', family='cash', ma=ma, vol=vol))
        rows.extend([
            dict(id=f'split60_ma{ma}_v15', family='split', ma=ma, vol=.15, red=.6, gold=.35),
            dict(id=f'rotate95_ma{ma}_v15', family='rotate', ma=ma, vol=.15)])
    rows.extend([
        dict(id='split80_ma200_v15', family='split', ma=200, vol=.15, red=.8, gold=.15),
        dict(id='rotate95_ma200_no_vol', family='rotate', ma=200, vol=None),
        dict(id='static60', family='static', ma=200, vol=None, red=.6, gold=.35),
        dict(id='gold95_ma200_v15', family='gold_trend', ma=200, vol=.15),
        dict(id='gold95_hold', family='gold_hold', ma=200, vol=None),
        dict(id='red95_hold', family='red_hold', ma=200, vol=None)])
    assert len(rows) == 18
    return rows


def protected_files():
    return [PROTOCOL, Path(__file__), INDEX, DB, ETF, OLD / 'comparison.json',
            ROOT / 'output/strategies/current.json', ROOT / 'strategy_registry.json',
            ROOT / 'strategy_corporate_actions.json',
            ROOT / 'output/valuation_bands/dividend_comparison_20261109/protocol.json']


def register():
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT / 'output/dividend_trend/runs' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    write_json(directory / 'registration.json', {
        'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'protocol': protocol, 'candidates': candidates(),
        'hashes': {str(p.relative_to(ROOT)): digest(p) for p in protected_files()},
        'investment_admission': False, 'production_changed': False})
    todo_write(directory, 'data')
    print(str(directory))


def fetch_warmup(job):
    year, directory, start, end = job
    path = directory / f'H20269_warmup_{year}.json'
    meta = path.with_suffix('.metadata.json')
    if not path.exists():
        session = requests.Session()
        session.trust_env = False
        response = session.get(ENDPOINT, params={'indexCode': 'H20269',
            'startDate': max(start, f'{year}-01-01').replace('-', ''),
            'endDate': min(end, f'{year}-12-31').replace('-', '')}, timeout=(8, 30))
        response.raise_for_status()
        path.write_bytes(response.content)
        write_json(meta, {'url': response.url, 'sha256': digest(path),
                         'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
    assert json.loads(meta.read_text(encoding='utf-8'))['sha256'] == digest(path)
    body = json.loads(path.read_text(encoding='utf-8'))
    assert str(body['code']) == '200' and body['data'], f'官方预热数据缺失: {year}'
    frame = pd.DataFrame(body['data'])
    assert frame.indexCode.eq('H20269').all()
    assert frame.indexNameCnAll.eq('中证红利低波动全收益指数').all()
    frame['date'] = pd.to_datetime(frame.tradeDate, format='%Y%m%d')
    return frame.set_index('date').close.astype(float)


def prepare_data(directory, protocol):
    raw = directory / 'raw'
    raw.mkdir(exist_ok=True)
    warmup_start, start, end = (protocol[k] for k in ('warmup_start', 'start', 'end'))
    cutoff = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    with ThreadPoolExecutor(max_workers=3) as pool:
        warm = list(pool.map(fetch_warmup, [(y, raw, warmup_start, cutoff)
                         for y in range(int(warmup_start[:4]), int(start[:4]) + 1)]))
    current = pd.read_csv(INDEX, parse_dates=['date']).set_index('date').close.astype(float)
    current.index = pd.DatetimeIndex(current.index).as_unit('ns')
    red = pd.concat(warm + [current]).sort_index()
    assert not red.index.has_duplicates
    calendar = build_calendar(warmup_start, end)
    excluded = red[~red.index.isin(calendar)]
    red = red.loc[red.index.isin(calendar)]
    assert red.index.equals(calendar), f'红利指数缺日: {calendar.difference(red.index)}'
    with duckdb.connect(str(DB), read_only=True) as con:
        gold = con.execute("SELECT b.date,b.open,b.close,a.adj_factor,b.source FROM bars_raw b "
            "JOIN adj_factor a USING(symbol,date) WHERE b.symbol='SH518880' AND b.date BETWEEN ? AND ? "
            "ORDER BY b.date", [warmup_start, end]).df().set_index('date')
    gold.index = pd.DatetimeIndex(gold.index).as_unit('ns')
    assert gold.index.equals(calendar), f'黄金缺日: {calendar.difference(gold.index)}'
    gold.to_csv(directory / 'gold_input.csv')
    prices = pd.DataFrame({'red': red, 'gold': gold.close * gold.adj_factor})
    assert np.isfinite(prices).all().all() and prices.gt(0).all().all()
    prices.to_csv(directory / 'index_gold_prices.csv', index_label='date')
    excluded.to_csv(directory / 'excluded_non_sessions.csv')
    etf = pd.read_csv(ETF, parse_dates=['trade_date'])
    etf = etf[etf.symbol.eq('SH512890')].set_index('trade_date').sort_index().loc[:end]
    etf.index = pd.DatetimeIndex(etf.index).as_unit('ns')
    assert etf.index[0] == pd.Timestamp('2019-01-18') and not etf.index.has_duplicates
    expected = calendar[calendar >= etf.index[0]]
    missing = expected.difference(etf.index)
    assert missing.equals(pd.DatetimeIndex(['2021-10-22'])), missing
    # 拆分停牌当日只估值，不允许按虚构价格成交；此后价格乘2保持等值单位连续。
    adjusted = etf.close * np.where(etf.index >= pd.Timestamp('2021-10-22'), 2., 1.)
    etf_prices = pd.DataFrame({'red': adjusted.reindex(expected).ffill(limit=1),
                             'gold': prices.gold.reindex(expected)})
    assert np.isfinite(etf_prices).all().all()
    etf_prices.to_csv(directory / 'etf_gold_prices.csv', index_label='date')
    write_json(directory / 'data_manifest.json', {
        'warmup_observations': int((calendar < start).sum()),
        'index_observations': len(prices.loc[start:end]), 'etf_observations': len(etf_prices),
        'index_name': '中证红利低波动全收益指数', 'index_code': 'H20269',
        'etf_listing_date': '2019-01-18', 'etf_no_fill_dates': ['2021-10-22'],
        'etf_corporate_action_evidence': 'output/valuation_bands/dividend_comparison_20261109/',
        'gold_basis': 'SH518880原始收盘*当日复权因子；全收益价格代理，基金费用已内含',
        'calendar': 'XSHG完整交易日，不填指数缺值，剔除官方接口非交易日占位',
        'hashes': {str(p.relative_to(directory)): digest(p) for p in
                   [directory / 'index_gold_prices.csv', directory / 'gold_input.csv',
                    directory / 'etf_gold_prices.csv'] + sorted(raw.glob('*'))}})
    return prices, etf_prices


def signal_targets(prices, candidate, vol_window=60):
    ma = prices.rolling(candidate['ma'], min_periods=candidate['ma']).mean()
    active = prices.gt(ma) & ma.notna()
    target = pd.DataFrame(0., index=prices.index, columns=prices.columns)
    family = candidate['family']
    if family == 'cash':
        target['red'] = .95 * active.red
    elif family == 'split':
        target['red'], target['gold'] = candidate['red'] * active.red, candidate['gold'] * active.gold
    elif family == 'rotate':
        target['red'] = .95 * active.red
        target['gold'] = .95 * (~active.red & active.gold)
    elif family == 'static':
        target['red'], target['gold'] = candidate['red'], candidate['gold']
    elif family == 'gold_trend':
        target['gold'] = .95 * active.gold
    elif family == 'gold_hold':
        target['gold'] = .95
    elif family == 'red_hold':
        target['red'] = .95
    else:
        raise ValueError(family)
    returns = prices.pct_change(fill_method=None)
    if candidate['vol'] is not None:
        variance = pd.Series(0., index=prices.index)
        for a in prices:
            for b in prices:
                variance += target[a] * target[b] * returns[a].rolling(vol_window).cov(returns[b]) * 252
        risk = variance.clip(lower=0).pow(.5)
        scale = (candidate['vol'] / risk.replace(0, np.nan)).clip(upper=1)
        scale = scale.where(risk.ne(0), 1.).fillna(0)
        target = target.mul(scale, axis=0)
    assert target.ge(0).all().all() and target.sum(axis=1).le(.95000000001).all()
    return target


def rebalance(values, cash, target, rate, minimum_fee, band=.03):
    """先卖后买，自融资；返回金额持仓、现金、成交与费用。"""
    values = np.asarray(values, dtype=float).copy()
    target = np.asarray(target, dtype=float).copy()
    nav = cash + values.sum()
    current = values / nav
    keep = (current > 0) & (target > 0) & (np.abs(current - target) < band)
    target[keep] = current[keep]
    if target.sum() > .95:
        target *= .95 / target.sum()
    wanted = target * nav
    trades, costs = [], 0.
    for j in np.flatnonzero(wanted < values - 1e-8):
        amount = values[j] - wanted[j]
        fee = max(minimum_fee, amount * rate)
        values[j] -= amount
        cash += amount - fee
        costs += fee
        trades.append((int(j), 'sell', float(amount), float(fee)))
    needed = np.maximum(wanted - values, 0.)
    count = int(np.count_nonzero(needed > 1e-8))
    available = max(0., cash - minimum_fee * count)
    if needed.sum() * (1 + rate) > available:
        needed *= available / (needed.sum() * (1 + rate)) if needed.sum() else 0.
    for j in np.flatnonzero(needed > 1e-8):
        amount = needed[j]
        fee = max(minimum_fee, amount * rate)
        assert cash + 1e-7 >= amount + fee
        values[j] += amount
        cash -= amount + fee
        costs += fee
        trades.append((int(j), 'buy', float(amount), float(fee)))
    assert cash >= -1e-7 and np.all(values >= -1e-7)
    assert abs(values.sum() + cash - (nav - costs)) < 1e-7
    return values, max(0., cash), trades, costs


def replay(prices, targets, start, capital=100000., cost=.001, min_fee=5.,
           delay=1, haircut=.006, band=.03, week_end=False, no_fill_dates=()):
    path = prices.loc[start:]
    returns = prices.pct_change(fill_method=None).loc[path.index].fillna(0).to_numpy()
    target_values = targets.reindex(path.index).to_numpy()
    periods = path.index.to_period('W-SUN')
    mask = np.r_[periods[:-1] != periods[1:], True] if week_end else np.r_[True, periods[1:] != periods[:-1]]
    cash, values = capital, np.zeros(2)
    queue, records, trades = {}, [], []
    forbidden = set(pd.to_datetime(list(no_fill_dates)))
    for i, day in enumerate(path.index):
        prior_nav = float(cash + values.sum())
        pnl = 0.
        if i:
            change = values * returns[i]
            pnl = float(change.sum())
            values += change
        days = (day - path.index[i-1]).days if i else 0
        management = float(values[0] * (1 - np.exp(-haircut * days / 365.2425)))
        values[0] -= management
        cost_today = 0.
        if i in queue:
            signal_day, target = queue.pop(i)
            if day in forbidden:
                queue[i+1] = (signal_day, target)
            else:
                values, cash, fills, cost_today = rebalance(values, cash, target, cost, min_fee, band)
                trades.extend({'date': str(day.date()), 'signal_date': str(signal_day.date()),
                               'asset': ('red', 'gold')[j], 'side': side, 'notional': amount,
                               'cost': fee} for j, side, amount, fee in fills)
        nav = float(cash + values.sum())
        records.append({'date': day, 'nav': nav, 'cash': cash, 'red_value': values[0],
                        'gold_value': values[1], 'pnl': pnl, 'trading_cost': cost_today,
                        'replication_cost': management, 'prior_nav': prior_nav,
                        'signal_today': bool(mask[i])})
        if mask[i] and i + delay < len(path):
            queue[i+delay] = (day, target_values[i].copy())
    daily = pd.DataFrame(records).set_index('date')
    orders = pd.DataFrame(trades, columns=['date', 'signal_date', 'asset', 'side', 'notional', 'cost'])
    return daily, orders


def metrics(nav):
    nav = nav.astype(float)
    assert not nav.index.has_duplicates and np.isfinite(nav).all() and nav.gt(0).all()
    annual = []
    for year, group in nav.groupby(nav.index.year):
        before = nav[nav.index < pd.Timestamp(year, 1, 1)]
        complete = (len(before) > 0 and nav.index[-1] >= build_calendar(f'{year}-01-01', f'{year}-12-31')[-1])
        annual.append({'year': int(year), 'return': float(group.iloc[-1] / (before.iloc[-1] if len(before) else nav.iloc[0]) - 1),
                       'complete': bool(complete)})
    full = [a for a in annual if a['complete']]
    worst = min(full, key=lambda a: a['return']) if full else None
    returns = nav.pct_change(fill_method=None).dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.2425
    dd = 1 - nav / nav.cummax()
    trough = dd.idxmax()
    return {'start': str(nav.index[0].date()), 'end': str(nav.index[-1].date()),
        'final_nav': float(nav.iloc[-1]), 'cagr': float((nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1),
        'max_drawdown': float(dd.max()), 'worst_full_year': worst['return'] if worst else None,
        'worst_year': worst['year'] if worst else None, 'annual': annual,
        'annual_volatility': float(returns.std() * np.sqrt(252)),
        'sharpe_zero_cash': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.,
        'drawdown_peak': str(nav.loc[:trough].idxmax().date()), 'drawdown_trough': str(trough.date()),
        'worst_252_sessions': float((nav / nav.shift(252) - 1).min()) if len(nav) > 252 else None}


def save_case(directory, sid, daily, trades, benchmark):
    folder = directory / 'cases' / sid
    folder.mkdir(parents=True, exist_ok=False)
    daily.to_csv(folder / 'daily.csv', index_label='date')
    trades.to_csv(folder / 'trades.csv', index=False)
    row = {'id': sid, **metrics(daily.nav), 'trade_count': len(trades),
        'trade_cost': float(daily.trading_cost.sum()), 'replication_cost': float(daily.replication_cost.sum()),
        'annual_one_way_turnover': float(sum(trades.notional) / daily.nav.mean() /
                   ((daily.index[-1] - daily.index[0]).days / 365.2425)),
        'average_red_weight': float((daily.red_value / daily.nav).mean()),
        'average_gold_weight': float((daily.gold_value / daily.nav).mean()),
        'average_cash_weight': float((daily.cash / daily.nav).mean())}
    row['dominates_red_benchmark'] = bool(row['cagr'] > benchmark['cagr'] and
        row['max_drawdown'] < benchmark['max_drawdown'] and row['worst_full_year'] > benchmark['worst_full_year'])
    row['personal_goals'] = {'cagr': row['cagr'] >= .135, 'worst_complete_year': row['worst_full_year'] >= -.05,
                             'max_drawdown': row['max_drawdown'] <= .40}
    write_json(folder / 'metrics.json', row)
    return row


def verify_cases(directory, prices, registered, rows):
    maximum_ledger_error, maximum_metric_error = 0., 0.
    for row in rows:
        folder = directory / 'cases' / row['id']
        daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
        daily.index = pd.DatetimeIndex(daily.index).as_unit('ns')
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
        error = daily.nav - (daily.prior_nav + daily.pnl - daily.trading_cost - daily.replication_cost)
        maximum_ledger_error = max(maximum_ledger_error, float(error.abs().max()))
        assert daily.index.equals(build_calendar(daily.index[0], daily.index[-1]))
        assert np.allclose(daily.cash + daily.red_value + daily.gold_value, daily.nav, rtol=0, atol=1e-7)
        assert daily.cash.ge(-1e-7).all() and daily[['red_value', 'gold_value']].ge(-1e-7).all().all()
        if len(trades):
            assert (trades.signal_date < trades.date).all()
            assert abs(trades.cost.sum() - daily.trading_cost.sum()) < 1e-7
        # 与主统计不同的逐日单遍峰值与年末比值计算。
        peak, largest = float(daily.nav.iloc[0]), 0.
        for value in daily.nav:
            peak = max(peak, float(value))
            largest = max(largest, (peak-value)/peak)
        year_end = daily.nav.groupby(daily.index.year).last()
        full = year_end.pct_change(fill_method=None).dropna()
        full = full.loc[full.index < daily.index[-1].year]
        cagr = np.expm1(np.log(daily.nav.iloc[-1] / daily.nav.iloc[0]) * 365.2425 /
                         (daily.index[-1] - daily.index[0]).days)
        diff = max(abs(cagr-row['cagr']), abs(largest-row['max_drawdown']),
                   abs(float(full.min())-row['worst_full_year']))
        maximum_metric_error = max(maximum_metric_error, diff)
        assert diff < 1e-12
    # 截断输入后信号与账户前缀应完全相同，防止居中窗口或未来信号。
    prefix_errors = {}
    for candidate in registered['candidates']:
        if candidate['id'] not in registered['protocol']['anchors']:
            continue
        prefix = prices.loc[:'2021-12-31']
        targets = signal_targets(prefix, candidate)
        full_targets = signal_targets(prices, candidate).loc[prefix.index]
        pd.testing.assert_frame_equal(targets, full_targets)
        daily, _ = replay(prefix, targets, registered['protocol']['start'])
        original = pd.read_csv(directory / 'cases' / candidate['id'] / 'daily.csv',
                               parse_dates=['date']).set_index('date').loc[daily.index]
        error = float((daily.nav - original.nav).abs().max())
        assert error < 1e-7
        prefix_errors[candidate['id']] = error
    current_hashes = {p: digest(ROOT / p) for p in registered['hashes']}
    assert current_hashes == registered['hashes'], '冻结输入或正式状态发生变化'
    assert maximum_ledger_error < 1e-7
    return {'status': 'passed', 'case_count': len(rows), 'independent_metric_max_error': maximum_metric_error,
        'self_financing_max_error': maximum_ledger_error, 'causal_prefix_max_errors': prefix_errors,
        'calendar_complete': True, 'no_same_day_fills': True, 'protected_inputs_unchanged': True,
        'lot_rounding_and_broker_fills_verified': False, 'investment_admission': False}


def run(directory):
    directory = Path(directory).resolve()
    assert not (directory / 'results.json').exists(), '已完成run不得覆盖；请重新登记'
    registered = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registered['hashes'].items():
        assert digest(ROOT / path) == expected, f'冻结输入已变: {path}'
    protocol = registered['protocol']
    prices, etf_prices = prepare_data(directory, protocol)
    start = protocol['start']
    red_nav = protocol['initial_capital'] * prices.red.loc[start:] / prices.red.loc[start:].iloc[0]
    baseline = metrics(red_nav)
    old = json.loads((OLD / 'comparison.json').read_text(encoding='utf-8'))
    prior = next(r for r in old['rows'] if r['strategy_id'] == 'dividend_lowvol_H20269_total_return')
    for field in ('final_nav', 'cagr', 'max_drawdown', 'worst_full_year'):
        assert abs(baseline[field] - prior[field]) < 1e-7, field
    red_nav.to_csv(directory / 'red_benchmark_nav.csv', index_label='date')
    todo_write(directory, 'replay')
    rows, paths, subset = [], {}, []
    scenarios = {'cost2': {'cost': .002}, 'delay2': {'delay': 2},
                 'fee1pct': {'haircut': .01}, 'week_end': {'week_end': True}}
    for candidate in registered['candidates']:
        sid = candidate['id']
        targets = signal_targets(prices, candidate)
        daily, trades = replay(prices, targets, start)
        rows.append(save_case(directory, sid, daily, trades, baseline))
        paths[sid] = daily.nav
        if sid not in protocol['anchors']:
            continue
        for name, arguments in scenarios.items():
            d, t = replay(prices, targets, start, **arguments)
            rows.append(save_case(directory, f'{sid}__{name}', d, t, baseline))
        subset_start = '2019-01-18'
        passive = metrics(100000 * etf_prices.red / etf_prices.red.iloc[0])
        for name, arguments in {'base': {}, 'cost2': {'cost': .002}, 'delay2': {'delay': 2}}.items():
            d, t = replay(etf_prices, targets, subset_start, haircut=0,
                          no_fill_dates=['2021-10-22'], **arguments)
            row = save_case(directory, f'{sid}__etf_{name}', d, t, passive)
            subset.append(row)
            rows.append(row)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('annual', 'personal_goals')}
                  for r in rows]).to_csv(directory / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'id': r['id'], **a} for r in rows for a in r['annual']]).to_csv(
                  directory / 'annual_returns.csv', index=False, encoding='utf-8-sig')
    # 分段延续账户，上一年末为初值，避免漏掉首个交易日收益。
    segments = []
    for sid, nav in {**paths, 'red_benchmark': red_nav}.items():
        for begin, end in [('2017-01-01', '2021-12-31'), ('2022-01-01', '2025-12-31'),
                           ('2017-01-01', '2024-12-31')]:
            prior_date = nav.index[nav.index < begin][-1]
            curve = nav.loc[prior_date:end]
            segments.append({'id': sid, 'window': begin[:4]+'-'+end[:4], **metrics(curve)})
    pd.DataFrame([{k: v for k, v in r.items() if k != 'annual'} for r in segments]).to_csv(
        directory / 'segments.csv', index=False, encoding='utf-8-sig')
    todo_write(directory, 'validation')
    verification = verify_cases(directory, prices, registered, rows)
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'results.json', {'protocol': protocol['version'], 'benchmark': baseline,
        'etf_benchmark': passive, 'rows': rows, 'segments': segments,
        'investment_admission': False, 'production_changed': False})
    todo_write(directory, 'delivery')
    print(json.dumps({'directory': str(directory), 'benchmark': baseline,
        'main_rows': [{k: v for k, v in r.items() if k != 'annual'} for r in rows
                      if '__' not in r['id']], 'verification': verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.register:
        register()
    elif args.run:
        run(args.run)
    else:
        parser.error('需要 --register 或 --run')
