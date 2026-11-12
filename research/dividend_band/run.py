"""红利低波估值带研究：登记、回放、压力与交付。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from research.dividend_trend.run import ROOT, digest, write_json, metrics
from research.dividend_band.engine import replay
from research.dividend_band.signals import indicators, target_weights
from research.dividend_band import data as data_module

PROTOCOL = Path(__file__).with_name('protocol.json')
DATA_DEFAULT = ROOT / 'output/dividend_band/data_20261111'
PRIOR_BENCHMARK = {'final_nav': 245623.36157056037, 'cagr': 0.09263010748803802,
                   'max_drawdown': 0.27527457797272903, 'worst_full_year': -0.16394905259139902}


def protected_files(data_directory):
    return [PROTOCOL, Path(__file__), Path(__file__).with_name('signals.py'),
            Path(__file__).with_name('engine.py'), Path(__file__).with_name('data.py'),
            Path(__file__).with_name('verify.py'),
            ROOT / 'research/dividend_trend/run.py',
            ROOT / 'research/valuation_bands/data.py',
            data_directory / 'manifest.json', data_directory / 'panel.csv',
            data_directory / 'etf_panel.csv',
            ROOT / 'output/strategies/current.json', ROOT / 'strategy_registry.json']


def save_case(directory, sid, daily, trades, benchmark):
    folder = directory / 'cases' / sid
    folder.mkdir(parents=True, exist_ok=False)
    daily.to_csv(folder / 'daily.csv', index_label='date')
    trades.to_csv(folder / 'trades.csv', index=False)
    years = (daily.index[-1] - daily.index[0]).days / 365.2425
    row = {'id': sid, **metrics(daily.nav), 'trade_count': int(len(trades)),
           'trade_cost': float(daily.trading_cost.sum()),
           'replication_cost': float(daily.replication_cost.sum()),
           'annual_one_way_turnover': float(trades.notional.sum() / daily.nav.mean() / years),
           'average_red_weight': float((daily.red_value / daily.nav).mean()),
           'average_bond_weight': float((daily.bond_value / daily.nav).mean()),
           'average_cash_weight': float((daily.cash / daily.nav).mean())}
    row['dominates_red_benchmark'] = bool(
        row['cagr'] > benchmark['cagr'] and row['max_drawdown'] < benchmark['max_drawdown']
        and row['worst_full_year'] > benchmark['worst_full_year'])
    row['personal_goals'] = {'cagr': row['cagr'] >= .135,
                             'worst_complete_year': row['worst_full_year'] >= -.05,
                             'max_drawdown': row['max_drawdown'] <= .40}
    write_json(folder / 'metrics.json', row)
    return row


def register(data_directory):
    data_directory = data_directory.resolve()
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT / 'output/dividend_band/runs' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'registration.json', {
        'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'protocol': protocol, 'candidates': protocol['candidates'],
        'anchors': protocol['anchors'], 'data_directory': str(data_directory.relative_to(ROOT)),
        'hashes': {str(p.relative_to(ROOT)): digest(p) for p in protected_files(data_directory)},
        'sample_kind': 'post_exposure_historical_research',
        'future_unseen_holdout': False,
        'investment_admission': False, 'production_changed': False})
    write_json(directory / 'status.json', {'phase': 'registered',
                                          'tasks': [{'task': 'data', 'status': 'completed'},
                                                    {'task': 'replay', 'status': 'pending'},
                                                    {'task': 'validation', 'status': 'pending'},
                                                    {'task': 'delivery', 'status': 'pending'}]})
    print(str(directory))


def run(directory, data_directory=None):
    directory = Path(directory).resolve()
    assert not (directory / 'results.json').exists(), '已完成run不得覆盖；请重新登记'
    registered = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registered['hashes'].items():
        assert digest(ROOT / path) == expected, f'冻结输入已变: {path}'
    protocol = registered['protocol']
    panel, etf_panel, manifest = data_module.load(ROOT / registered['data_directory'])
    signals = indicators(panel)
    tables = protocol['band_tables']
    start, end = protocol['start'], protocol['end']

    indicator_path = directory / 'indicators.csv'
    signals.to_csv(indicator_path, index_label='date')
    signal_start = signals.loc[start:]
    assert signal_start.dividend_yield.notna().all(), '回放起点起股息率不得缺失'
    assert signal_start.dividend_pct.notna().all(), '回放起点起百分位不得缺失'

    red_nav = 100000. * panel.red.loc[start:] / panel.red.loc[start:].iloc[0]
    baseline = metrics(red_nav)
    for field, expected in PRIOR_BENCHMARK.items():
        assert abs(baseline[field] - expected) < 1e-6, (field, baseline[field], expected)
    red_nav.to_csv(directory / 'red_benchmark_nav.csv', index_label='date')
    write_json(directory / 'status.json', {'phase': 'replay', 'tasks': [
        {'task': 'data', 'status': 'completed'}, {'task': 'replay', 'status': 'in_progress'},
        {'task': 'validation', 'status': 'pending'}, {'task': 'delivery', 'status': 'pending'}]})

    scenarios = {'cost2': {'cost': .002}, 'cost3': {'cost': .003},
                 'delay2': {'delay': 2}, 'month': {'freq': 'M'}}
    rows, paths = [], {}
    for candidate in protocol['candidates']:
        targets = target_weights(panel, signals, candidate, tables)
        daily, trades = replay(panel[['red', 'bond']], targets, start,
                               no_fill_dates=['2018-08-17'])
        row = save_case(directory, candidate['id'], daily, trades, baseline)
        rows.append(row)
        paths[candidate['id']] = daily.nav
        if candidate['id'] not in protocol['anchors']:
            continue
        for name, arguments in scenarios.items():
            d, t = replay(panel[['red', 'bond']], targets, start,
                          no_fill_dates=['2018-08-17'], **arguments)
            rows.append(save_case(directory, f"{candidate['id']}__{name}", d, t, baseline))

    # 上市后ETF子样本：红利改用512890真实价格，不再扣指数复制费。
    passive = metrics(100000. * etf_panel.red / etf_panel.red.iloc[0])
    for candidate_id in protocol['anchors']:
        candidate = next(c for c in protocol['candidates'] if c['id'] == candidate_id)
        targets = target_weights(panel, signals, candidate, tables)
        etf_targets = targets.reindex(etf_panel.index)
        assert etf_targets.notna().all().all()
        etf_targets = etf_targets.copy()
        etf_targets.loc[~etf_panel.bond_listed.astype(bool), 'bond'] = 0.
        daily, trades = replay(etf_panel[['red', 'bond']], etf_targets, '2019-01-18',
                               haircut=0., no_fill_dates=['2021-10-22'])
        rows.append(save_case(directory, f'{candidate_id}__etf', daily, trades, passive))

    pd.DataFrame([{k: v for k, v in r.items() if k not in ('annual', 'personal_goals')}
                  for r in rows]).to_csv(directory / 'comparison.csv', index=False,
                                         encoding='utf-8-sig')
    pd.DataFrame([{'id': r['id'], **a} for r in rows for a in r['annual']]).to_csv(
        directory / 'annual_returns.csv', index=False, encoding='utf-8-sig')

    segments = []
    for sid, nav in {**paths, 'red_benchmark': red_nav}.items():
        for begin, finish in [('2017-01-01', '2021-12-31'), ('2022-01-01', '2025-12-31'),
                              ('2017-01-01', '2024-12-31')]:
            prior_date = nav.index[nav.index < begin][-1]
            curve = nav.loc[prior_date:finish]
            segments.append({'id': sid, 'window': begin[:4] + '-' + finish[:4],
                             **metrics(curve)})
    pd.DataFrame([{k: v for k, v in r.items() if k != 'annual'} for r in segments]).to_csv(
        directory / 'segments.csv', index=False, encoding='utf-8-sig')

    write_json(directory / 'status.json', {'phase': 'validation', 'tasks': [
        {'task': 'data', 'status': 'completed'}, {'task': 'replay', 'status': 'completed'},
        {'task': 'validation', 'status': 'in_progress'}, {'task': 'delivery', 'status': 'pending'}]})

    from research.dividend_band.verify import verify_run
    verification = verify_run(directory, panel, etf_panel, signals, registered, rows)
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'results.json', {
        'protocol': protocol['version'], 'benchmark': baseline, 'etf_benchmark': passive,
        'rows': rows, 'segments': segments, 'data_manifest': manifest,
        'investment_admission': False, 'production_changed': False})
    write_json(directory / 'status.json', {'phase': 'complete', 'tasks': [
        {'task': 'data', 'status': 'completed'}, {'task': 'replay', 'status': 'completed'},
        {'task': 'validation', 'status': 'completed'},
        {'task': 'delivery', 'status': 'completed'}]})
    print(json.dumps({'directory': str(directory), 'verification': verification,
                      'main_rows': [{k: v for k, v in r.items() if k != 'annual'}
                                    for r in rows if '__' not in r['id']]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--data', type=Path, default=DATA_DEFAULT)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.register:
        register(args.data)
    elif args.run:
        run(args.run)
    else:
        parser.error('需要--register或--run <目录>')
