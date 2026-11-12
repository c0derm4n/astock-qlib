"""对最终研究候选的独立信号、份额重建及已登记参数压力复核。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .run import ROOT, digest, write_json, metrics, replay, save_case, verify_cases
from .relative import signal_targets

SID = 'dual_blend_ma200_v15'
SCENARIOS = {
    'ma150': {'ma': 150}, 'ma250': {'ma': 250},
    'vol12': {'vol': .12}, 'vol18': {'vol': .18},
    'cost3': {'cost': .003}, 'band0': {'band': 0.}, 'band5': {'band': .05}}


def independent_targets(prices, ma=200, vol=.15):
    """直接逐窗计算均值和协方差，与pandas滚动信号实现独立。"""
    values = prices.to_numpy()
    target = np.zeros_like(values)
    returns = values[1:] / values[:-1] - 1
    for i in range(max(252, ma-1, 60), len(values)):
        momentum = sum(values[i] / values[i-n] - 1 for n in (63,126,252)) / 3
        eligible = (momentum > 0) & (values[i] > values[i-ma+1:i+1].mean(axis=0))
        if not np.any(eligible):
            continue
        ranked = np.where(eligible, momentum, -np.inf)
        winner = int(np.argmax(ranked))
        weights = np.zeros(2)
        weights[winner] = .95
        covariance = np.cov(returns[i-60:i], rowvar=False, ddof=1) * 252
        risk = np.sqrt(max(float(weights @ covariance @ weights), 0.))
        target[i] = weights * min(1., vol / risk) if risk > 0 else weights
    return pd.DataFrame(target, index=prices.index, columns=prices.columns)


def independent_units_check(prices, folder, haircut):
    """只读价格与逐笔交易，以份额*价格重建账户，不读取引擎pnl或持仓字段。"""
    daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
    trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date','signal_date'])
    daily.index = pd.DatetimeIndex(daily.index).as_unit('ns')
    observed = prices.loc[daily.index]
    groups = {day: group for day, group in trades.groupby('date')}
    cash, quantity = 100000., np.zeros(2)
    reconstructed = []
    for i, day in enumerate(daily.index):
        p = observed.loc[day].to_numpy()
        if i:
            days = (day-daily.index[i-1]).days
            quantity[0] *= np.exp(-haircut*days/365.2425)
        for row in groups.get(day, pd.DataFrame()).itertuples():
            asset = ('red','gold').index(row.asset)
            self_financing_sign = 1 if row.side == 'buy' else -1
            quantity[asset] += self_financing_sign * row.notional / p[asset]
            cash -= self_financing_sign * row.notional + row.cost
            assert abs(row.cost-max(5.,row.notional*.001)) < 1e-7
        reconstructed.append(cash+float(quantity@p))
        assert cash >= -1e-6 and quantity.min() >= -1e-6
    error = float(np.max(np.abs(np.asarray(reconstructed)-daily.nav.to_numpy())))
    assert error < 1e-6, error
    return error


def register(source):
    source = source.resolve()
    assert (source / 'results.json').exists()
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT / 'output/dividend_trend/validation' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), source/'results.json', source/'index_gold_prices.csv',
             source/'etf_gold_prices.csv', source/'registration.json',
             ROOT/'research/dividend_trend/run.py', ROOT/'research/dividend_trend/relative.py']
    write_json(directory / 'registration.json', {
        'source_run': str(source.relative_to(ROOT)), 'candidate': SID,
        'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'purpose': '在已查看两批历史后，仅复核blend候选，不依据本轮结果重新选择参数',
        'scenarios': SCENARIOS,
        'hashes': {str(p.relative_to(ROOT)):digest(p) for p in files},
        'independent_checks':['逐窗numpy信号','份额和现金重建指数账户','上市后ETF价格代理份额重建'],
        'future_unseen_holdout': False, 'investment_admission': False})
    print(str(directory))


def run(directory):
    directory = directory.resolve()
    assert not (directory/'results.json').exists()
    reg = json.loads((directory/'registration.json').read_text(encoding='utf-8'))
    source = ROOT / reg['source_run']
    for path, expected in reg['hashes'].items():
        assert digest(ROOT/path) == expected, path
    original = json.loads((source/'results.json').read_text(encoding='utf-8'))
    prices = pd.read_csv(source/'index_gold_prices.csv', parse_dates=['date']).set_index('date')
    prices.index = pd.DatetimeIndex(prices.index).as_unit('ns')
    etf = pd.read_csv(source/'etf_gold_prices.csv', parse_dates=['date']).set_index('date')
    etf.index = pd.DatetimeIndex(etf.index).as_unit('ns')
    independent = independent_targets(prices)
    primary = signal_targets(prices, dict(family='dual',lookback='blend'))
    error = float((independent-primary).abs().to_numpy().max())
    assert error < 1e-12, error
    index_error = independent_units_check(prices, source/'cases'/SID, .006)
    etf_error = independent_units_check(etf, source/'cases'/f'{SID}__etf_base', 0.)
    rows = []
    for name, arguments in reg['scenarios'].items():
        target_args = {k:v for k,v in arguments.items() if k in ('ma','vol')}
        replay_args = {k:v for k,v in arguments.items() if k not in ('ma','vol')}
        target = independent_targets(prices, **target_args)
        daily,trades = replay(prices,target,'2016-09-12',**replay_args)
        rows.append(save_case(directory,f'{SID}__{name}',daily,trades,original['benchmark']))
    verify = verify_cases(directory, prices, {**reg,'candidates':[]}, rows)
    verify.update(independent_signal_max_error=error, index_unit_reconstruction_max_error=index_error,
                  etf_unit_reconstruction_max_error=etf_error)
    write_json(directory/'verification.json',verify)
    write_json(directory/'results.json',{'source_run':reg['source_run'],'candidate':SID,'rows':rows})
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('annual','personal_goals')}
                  for r in rows]).to_csv(directory/'comparison.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({'directory':str(directory),'verification':verify,
        'rows':[{k:v for k,v in r.items() if k in ('id','cagr','max_drawdown','worst_full_year',
                  'dominates_red_benchmark')} for r in rows]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',type=Path)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.register:
        register(args.register)
    elif args.run:
        run(args.run)
    else:
        parser.error('需要--register <来源run>或--run <验证目录>')
