"""正式A策略：自融资指数等值账户，周首信号、T+1收盘执行。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path

import requests
from src.artifacts import write_json_atomic, json_safe
from src.runs import sha256
from src.valuation.data import build_calendar

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


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / 'data/formal/dividend_band/seed/panel.csv'
ENDPOINT = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
WARMUP = '2014-01-02'
SID = 'dividend_band_a'
PREVIEW_REASON = 'A采用指数等值与国债复权收益模拟；不能将等值仓位当作ETF整数份额委托，暂不支持真实账户试算'
BANDS = [[5.0, .95], [4.8, .85], [4.2, .70], [4.0, .60], [3.8, .50], [3.6, .35], [-999., .20]]
NAMES = {'red': '红利低波（指数等值）', 'bond': '十年国债（复权等值）'}
SYMBOLS = {'red': 'H20269', 'bond': 'SH511260'}


def validate_panel(frame, end):
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).as_unit('ns')
    frame = frame.sort_index().loc[WARMUP:end]
    if frame.index.has_duplicates or not frame.index.equals(build_calendar(WARMUP, end)):
        raise ValueError('A策略价格与全收益指数必须覆盖完整交易日和252日预热，禁止填补缺口')
    if not np.isfinite(frame[['red', 'price', 'bond']]).all().all() or not frame[['red', 'price', 'bond']].gt(0).all().all():
        raise ValueError('A策略指数或国债估值缺失/非正')
    if frame.bond_listed.dtype != bool:
        raise ValueError('A策略国债上市标志必须为布尔值')
    if not frame.bond_listed.equals(pd.Series(frame.index >= pd.Timestamp('2017-08-24'), index=frame.index, name='bond_listed')):
        raise ValueError('国债上市前不可持有，不得伪造上市前收益')
    frame.index.name = 'date'
    return frame


def _read_panel(path):
    return pd.read_csv(path, index_col='date', parse_dates=True)


def _extend_panel(frame, market, destination, fetch):
    end = market.calendar[-1]
    calendar = build_calendar(WARMUP, end)
    frame = frame.loc[:end].copy()
    first_year = frame.index[-1].year
    raw_dir = destination / 'raw'
    sources = []
    if fetch:
        raw_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.trust_env = False
        for column, code, name in [
            ('red', 'H20269', '中证红利低波动全收益指数'),
            ('price', 'H30269', '中证红利低波动指数')]:
            for year in range(first_year, end.year + 1):
                response = session.get(ENDPOINT, params={'indexCode': code, 'startDate': f'{year}0101',
                    'endDate': min(end, pd.Timestamp(year, 12, 31)).strftime('%Y%m%d')}, timeout=(8, 30))
                response.raise_for_status()
                body = response.json()
                if str(body.get('code')) != '200' or not body.get('data'):
                    raise ValueError(f'{code}官方行情响应无效')
                incoming = pd.DataFrame(body['data'])
                if not incoming.indexCode.eq(code).all() or not incoming.indexNameCnAll.eq(name).all():
                    raise ValueError(f'{code}指数身份不符合正式定义')
                incoming.index = pd.to_datetime(incoming.tradeDate, format='%Y%m%d')
                if incoming.index.has_duplicates:
                    raise ValueError(f'{code}官方行情日期重复')
                values = pd.to_numeric(incoming.close)
                values = values[values.index.isin(calendar)]
                overlap = frame.index.intersection(values.index)
                old = frame.loc[overlap, column].dropna()
                if not np.allclose(old, values.loc[old.index], rtol=0, atol=.000001):
                    raise ValueError(f'{code}历史发生修订，须核查后新登记，不静默改写')
                frame = frame.reindex(frame.index.union(values.index))
                frame.loc[values.index, column] = values
                path = raw_dir / f'{code}_{year}.json'
                path.write_bytes(response.content)
                sources.append({'url': response.url, 'file': str(path.relative_to(destination)), 'sha256': sha256(path)})
    frame = frame.reindex(calendar)
    close = market.wide('close')['SH511260']
    factor = market.wide('adj_factor')['SH511260'].ffill()
    bond = (close * factor).reindex(calendar)
    observed = frame.bond.notna() & bond.notna()
    observed &= frame.index >= pd.Timestamp('2017-08-24')
    if not np.allclose(frame.loc[observed, 'bond'], bond.loc[observed], rtol=0, atol=.000001):
        raise ValueError('国债复权历史与登记种子不一致，拒绝静默替换')
    frame['bond'] = frame.bond.fillna(bond)
    frame['bond_listed'] = frame.index >= pd.Timestamp('2017-08-24')
    frame.loc[~frame.bond_listed, 'bond'] = 1.
    return validate_panel(frame, end), sources


def initialize_seed(source, market):
    if SEED.exists() or SEED.with_suffix('.metadata.json').exists():
        raise ValueError('正式A策略种子已存在，不覆盖')
    source = Path(source).resolve()
    manifest = json.loads(source.with_name('manifest.json').read_text(encoding='utf-8'))
    if sha256(source) != manifest['files']['panel.csv']:
        raise ValueError('研究种子哈希不匹配')
    frame = _read_panel(source)[['red', 'price', 'bond', 'bond_listed']]
    SEED.parent.mkdir(parents=True, exist_ok=True)
    frame, sources = _extend_panel(frame, market, SEED.parent, True)
    frame.to_csv(SEED, index_label='date')
    write_json_atomic(SEED.with_suffix('.metadata.json'), {
        'kind': 'dividend_band', 'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source),
        'sha256': sha256(SEED), 'warmup_start': WARMUP, 'end': str(frame.index[-1].date()),
        'sources': sources, 'bond_gap': '2018-08-17以已核查基金净值102.057仅作估值，不成交',
        'valuation_basis': '252交易日历史隐含派息代理；非中证官方股息率，历史快照非PIT'})


def prepare_data(directory, market, fetch=False, previous_directory=None):
    parent = Path(previous_directory) / 'dividend_data/panel.csv' if previous_directory else SEED
    if not parent.exists():
        parent = SEED
    if not parent.exists():
        raise ValueError('缺少正式A策略种子，请先运行 python -m src.formal_dividend --initialize-seed')
    if parent == SEED:
        metadata = json.loads(SEED.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        expected = metadata['sha256']
    else:
        completed = json.loads((Path(previous_directory) / 'completion.json').read_text(encoding='utf-8'))
        expected = completed['artifact_hashes']['dividend_data/panel.csv']
    if sha256(parent) != expected:
        raise ValueError('正式A策略输入快照哈希不匹配')
    destination = directory / 'dividend_data'
    destination.mkdir(exist_ok=False)
    frame, sources = _extend_panel(_read_panel(parent), market, destination, fetch)
    frame.to_csv(destination / 'panel.csv', index_label='date')
    write_json_atomic(destination / 'manifest.json', {'kind': 'dividend_band', 'parent_sha256': sha256(parent),
        'data_sha256': sha256(destination / 'panel.csv'), 'warmup_start': WARMUP,
        'end': str(frame.index[-1].date()), 'observations': len(frame), 'sources': sources,
        'valuation_basis': '隐含股息率代理，非官方股息率或PIT留存', 'bond_prelisting_cash': True,
        'bond_gap_valuation_only': '2018-08-17'})
    return frame


def signal_frame(panel, definition):
    policy = definition['policy']
    implied = panel.red / panel.red.shift(1) * panel.price.shift(1) - panel.price
    implied.iloc[0] = np.nan
    dy = implied.rolling(252, min_periods=252).sum() / panel.price * 100
    table = policy['band_table']
    if not table or any(table[i][0] <= table[i+1][0] for i in range(len(table)-1)):
        raise ValueError('A策略股息率档位必须降序排列')
    weights = pd.Series([next((float(w) for threshold, w in table if v >= threshold), np.nan)
                         if np.isfinite(v) else np.nan for v in dy], index=panel.index)
    red = weights.ffill().fillna(0.)
    bond = (.95 - red).where(panel.bond_listed, 0.)
    return pd.DataFrame({'dividend_yield_pct': dy, 'implied_yield': dy / 100,
        'red_target': red, 'bond_target': bond, 'cash_target': 1 - red - bond}, index=panel.index)


def signal_detail(row, date):
    return {'date': str(pd.Timestamp(date).date()), 'implied_yield': float(row.implied_yield),
            'red_target': float(row.red_target), 'bond_target': float(row.bond_target),
            'cash_target': float(row.cash_target), 'price_index': 'H30269', 'return_index': 'H20269'}


def holdings_rows(row):
    nav = float(row.nav)
    return [{'asset': a, 'symbol': SYMBOLS[a], 'code': SYMBOLS[a], 'name': NAMES[a],
             'value': float(row[a + '_value']), 'weight': float(row[a + '_value'] / nav),
             'quantity': None, 'price': None, 'equivalent': True} for a in ASSETS if row[a + '_value'] > 1e-8]


def allocation_advice(row, target, definition):
    values, cash, fills, _ = rebalance([row.red_value, row.bond_value], row.cash,
        [target.red_target, target.bond_target], definition['policy']['transaction_cost'],
        definition['policy']['minimum_fee'], definition['policy']['rebalance_band'])
    orders = {ASSETS[j]: (side, amount) for j, side, amount, fee in fills}
    rows = []
    for i, asset in enumerate(ASSETS):
        current = float(row[asset + '_value'] / row.nav)
        side, amount = orders.get(asset, ('hold', 0.))
        rows.append({'asset': asset, 'symbol': SYMBOLS[asset], 'name': NAMES[asset],
            'side': side, 'current_weight': current, 'target_weight': float(target[asset + '_target']),
            'estimated_amount': float(amount), 'quantity': None, 'reference_price': None,
            'reason': '隐含股息率档位＋5个百分点漂移带；金额为信号收盘的等值估算，次日收盘重新计价'})
    return rows


def verify_account(panel, daily, trades, capital, policy):
    """按份额、价格和现金独立重建，不复用市值递推。"""
    quantities = np.zeros(2)
    cash = float(capital)
    groups = {date: rows for date, rows in trades.groupby('date')}
    errors = []
    for i, date in enumerate(daily.index):
        if i:
            days = (date-daily.index[i-1]).days
            quantities[0] *= np.exp(-policy['replication_fee']*days/365.2425)
        prices = panel.loc[date, ['red','bond']].to_numpy(float)
        for trade in groups.get(str(date.date()), pd.DataFrame()).itertuples():
            j = ASSETS.index(trade.asset)
            sign = 1 if trade.side == 'buy' else -1
            if abs(trade.cost-max(policy['minimum_fee'],trade.notional*policy['transaction_cost']))>1e-7:
                raise ValueError('A策略成交费用独立核对失败')
            quantities[j] += sign*trade.notional/prices[j]
            cash -= sign*trade.notional+trade.cost
        if cash < -1e-7 or quantities.min() < -1e-7:
            raise ValueError('A策略独立重建出现负现金/份额')
        errors.append(abs(cash+quantities@prices-daily.loc[date,'nav']))
    maximum = float(max(errors))
    if maximum > 1e-6:
        raise ValueError('A策略独立份额与现金重建失败')
    return maximum


def build_strategy(directory, definition, spec, panel, run_id, stats_fn, next_session_fn):
    start, capital = spec['start'], spec['initial_capital']
    signals = signal_frame(panel, definition)
    if signals.loc[start:, 'implied_yield'].isna().any():
        raise ValueError('A策略252日派息预热不足')
    targets = signals[['red_target', 'bond_target']].rename(columns={'red_target':'red','bond_target':'bond'})
    policy = definition['policy']
    daily, trades = replay(panel[['red','bond']], targets, start, capital=capital,
        cost=policy['transaction_cost'], min_fee=policy['minimum_fee'], haircut=policy['replication_fee'],
        band=policy['rebalance_band'], no_fill_dates=['2018-08-17'])
    error = daily.nav - (daily.prior_nav + daily.pnl - daily.trading_cost - daily.replication_cost)
    if error.abs().max() > 1e-7 or daily.cash.min() < -1e-7:
        raise ValueError('A策略等值账本验收失败')
    unit_error = verify_account(panel, daily, trades, capital, policy)
    calendar = daily.index
    signal_days = calendar[period_mask(calendar, 'W')]
    if len(trades):
        s = pd.to_datetime(trades.signal_date)
        executed = pd.to_datetime(trades.date)
        if not s.isin(signal_days).all() or not np.array_equal(
                calendar[calendar.get_indexer(s)+1].to_numpy(), executed.to_numpy()):
            raise ValueError('A策略存在非周首或非T+1收盘成交')
    sid = definition['id']
    daily.to_csv(directory / f'{sid}_daily.csv', index_label='date')
    trades['symbol'] = trades.asset.map(SYMBOLS)
    trades['quantity'] = None
    trades['price'] = None
    trades['execution_basis'] = '指数等值/复权等值的收盘模拟'
    trades.to_csv(directory / f'{sid}_trades.csv', index=False)
    weekly_signals = signals.loc[signal_days].copy()
    weekly_signals.index.name = 'session'
    weekly_signals.to_csv(directory / f'{sid}_signals.csv')
    report = pd.DataFrame({'account':daily.nav,'cash':daily.cash,'value':daily.red_value+daily.bond_value})
    report.index.name = 'date'
    report.to_csv(directory / f'{sid}_report.csv')
    metrics, dd = stats_fn(report, capital)
    last_signal = signal_days[-1]
    signal = signals.loc[last_signal]
    signal_day = daily.loc[last_signal]
    current = daily.iloc[-1]
    detail = signal_detail(signal, last_signal)
    advice = allocation_advice(signal_day, signal, definition)
    weekly = [{'date': str(d.date()), 'account': float(daily.loc[d,'nav']), 'cash': float(daily.loc[d,'cash']),
               'dividend_receivable': 0., 'holdings': holdings_rows(daily.loc[d]),
               'dividend_signal': signal_detail(signals.loc[d], d)} for d in signal_days[-50:]][::-1]
    capabilities = {'account_preview': False, 'executable_orders': False, 'reason': PREVIEW_REASON}
    context = {'date': str(last_signal.date()), 'prices': {}, 'eligible': [],
               'capabilities': capabilities, 'dividend_signal': detail}
    payload = {'id':sid,'name':definition['name'],'color':definition['color'],'kind':'dividend_band',
        'status':'formal_service','description':definition['description'],'version':spec['version'],'run_id':run_id,
        'data_start':start,'data_end':str(calendar[-1].date()),'initial_capital':capital,
        'account_basis':definition['account_basis'],'metrics':metrics,'investment_admission':False,
        'goals':{'cagr':metrics['cagr']>=spec['goals']['cagr'],
                 'worst_complete_year':bool(metrics['worst_complete_year'] and metrics['worst_complete_year']['return']>=spec['goals']['worst_complete_year']),
                 'max_drawdown':metrics['max_drawdown']<=spec['goals']['max_drawdown']},
        'reference_only':False,'dividend_signal':detail,'cash':float(current.cash),'dividend_receivable':0.,
        'holdings':holdings_rows(current),'capabilities':capabilities,
        'limitations':['隐含股息率代理，非中证官方股息率；历史快照非PIT',
            '指数与复权等值账户，未验证ETF整手、容量或实际现金分红到账',
            '历史基础情景优于持有，成本和数据时点压力下优势可能消失；非未来收益保证',
            '正式服务登记不等于年化13.5%联合目标或投资准入通过'],
        'advice':{'signal_date':str(last_signal.date()),'execution_date':str(next_session_fn(last_signal).date()),
            'account':float(signal_day.nav),'cash':float(signal_day.cash),'rows':advice,
            'orders_count':sum(r['side']!='hold' for r in advice),'paper_only':True,
            'execution_basis':'next_session_close','basis':'模型账户信号收盘的等值金额；T+1收盘模拟执行，非券商委托',
            'decision_detail':f"隐含股息率{signal.implied_yield:.2%}；红利{signal.red_target:.0%}、国债{signal.bond_target:.0%}、现金{signal.cash_target:.0%}"},
        'series':{'dates':calendar.strftime('%Y-%m-%d').tolist(),'capital':daily.nav.tolist(),
            'drawdown':(-dd).tolist(),'cash':daily.cash.tolist()},
        'weekly_holdings':weekly,'weekly_allocation':weekly,
        'validation':{'weekly_t_plus_1':True,'execution_price':'close','trade_count':len(trades),'issues':[],
            'ledger':{'status':'passed','max_nav_error':max(float(error.abs().max()),unit_error),
                      'self_financing_max_error':float(error.abs().max()),'independent_units_max_error':unit_error},
            'account_model':'index_equivalent_fractional','lot_rounding_and_capacity':False}}
    write_json_atomic(directory / f'{sid}.json', json_safe(payload))
    write_json_atomic(directory / f'{sid}_context.json', context)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize-seed', action='store_true')
    args = parser.parse_args()
    if args.initialize_seed:
        from src.market import load_market
        initialize_seed(ROOT / 'output/dividend_band/data_20261111/panel.csv',
                        load_market(ROOT / 'data/formal/market.duckdb'))
        print('正式A策略种子已建立：' + str(SEED))
