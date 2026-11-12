"""领先候选的补充压力复核：看到主run结果后另立登记，不覆盖主协议与档位。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from research.dividend_trend.run import ROOT, digest, write_json
from research.dividend_band import data as data_module
from research.dividend_band.engine import replay
from research.dividend_band.run import save_case
from research.dividend_band.signals import indicators, target_weights
from research.dividend_band.verify import independent_units, single_pass_metrics

PROTOCOL = ROOT / 'research/dividend_band/protocol.json'
CANDIDATES = ['abs_band_wide', 'pct_band', 'erp_pct_band']
SCENARIOS = {'cost2': {'cost': .002}, 'cost3': {'cost': .003},
             'delay2': {'delay': 2}, 'month': {'freq': 'M'}}


def protected(parent, data_directory):
    return [PROTOCOL, ROOT / 'research/dividend_band/signals.py',
            ROOT / 'research/dividend_band/engine.py',
            parent / 'results.json', parent / 'comparison.csv',
            data_directory / 'manifest.json', data_directory / 'panel.csv',
            data_directory / 'etf_panel.csv',
            ROOT / 'output/strategies/current.json', ROOT / 'strategy_registry.json']


def register(parent, data_directory):
    parent, data_directory = parent.resolve(), data_directory.resolve()
    assert (parent / 'results.json').exists()
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    rows = pd.read_csv(parent / 'comparison.csv')
    known = {row.id: {'cagr': float(row.cagr), 'max_drawdown': float(row.max_drawdown),
                      'worst_full_year': float(row.worst_full_year)}
             for row in rows.itertuples()}
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT / 'output/dividend_band/runs' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'registration.json', {
        'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'purpose': '主run显示abs_band_wide、pct_band、erp_pct_band在全期三项优于基准；本批只对这三个已见结果补充成本与执行压力，不新增档位、不按结果更换主参数。',
        'parent_run': str(parent.relative_to(ROOT)),
        'data_directory': str(data_directory.relative_to(ROOT)),
        'candidates': CANDIDATES, 'scenarios': SCENARIOS,
        'known_at_selection': {k: known[k] for k in CANDIDATES if k in known},
        'protocol': protocol,
        'hashes': {str(p.relative_to(ROOT)): digest(p) for p in protected(parent, data_directory)},
        'sample_kind': 'post_selection_stress_only', 'future_unseen_holdout': False,
        'investment_admission': False, 'production_changed': False})
    write_json(directory / 'status.json', {'phase': 'replay', 'tasks': [
        {'task': 'replay', 'status': 'pending'},
        {'task': 'validation', 'status': 'pending'}]})
    print(str(directory))


def run(directory, parent):
    directory, parent = directory.resolve(), parent.resolve()
    assert not (directory / 'results.json').exists()
    registered = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registered['hashes'].items():
        assert digest(ROOT / path) == expected, f'冻结输入已变: {path}'
    protocol = registered['protocol']
    tables = protocol['band_tables']
    panel, etf_panel, _ = data_module.load(ROOT / registered['data_directory'])
    signals = indicators(panel)
    start = protocol['start']
    baseline = json.loads((parent / 'results.json').read_text(encoding='utf-8'))['benchmark']
    passive = json.loads((parent / 'results.json').read_text(encoding='utf-8'))['etf_benchmark']
    rows = []
    for candidate_id in registered['candidates']:
        candidate = next(c for c in protocol['candidates'] if c['id'] == candidate_id)
        targets = target_weights(panel, signals, candidate, tables)
        for name, arguments in {'base': {}, **SCENARIOS}.items():
            daily, trades = replay(panel[['red', 'bond']], targets, start,
                                   no_fill_dates=['2018-08-17'], **arguments)
            rows.append(save_case(directory, f'{candidate_id}__{name}', daily, trades, baseline))
        etf_targets = targets.reindex(etf_panel.index).copy()
        etf_targets.loc[~etf_panel.bond_listed.astype(bool), 'bond'] = 0.
        daily, trades = replay(etf_panel[['red', 'bond']], etf_targets, '2019-01-18',
                               haircut=0., no_fill_dates=['2021-10-22'])
        rows.append(save_case(directory, f'{candidate_id}__etf', daily, trades, passive))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('annual', 'personal_goals')}
                  for r in rows]).to_csv(directory / 'comparison.csv', index=False,
                                         encoding='utf-8-sig')
    pd.DataFrame([{'id': r['id'], **a} for r in rows for a in r['annual']]).to_csv(
        directory / 'annual_returns.csv', index=False, encoding='utf-8-sig')

    maximum_metric = 0.
    for row in rows:
        daily = pd.read_csv(directory / 'cases' / row['id'] / 'daily.csv',
                            parse_dates=['date']).set_index('date')
        daily.index = pd.DatetimeIndex(daily.index).as_unit('ns')
        independent = single_pass_metrics(daily.nav)
        for field in ('cagr', 'max_drawdown', 'worst_full_year'):
            maximum_metric = max(maximum_metric, abs(independent[field] - row[field]))
        error = daily.nav - (daily.prior_nav + daily.pnl - daily.trading_cost
                             - daily.replication_cost)
        assert float(error.abs().max()) < 1e-7
    assert maximum_metric < 1e-9
    unit = {}
    for candidate_id in registered['candidates']:
        unit[candidate_id] = independent_units(panel[['red', 'bond']],
                                              directory / 'cases' / f'{candidate_id}__base', .006)
        assert unit[candidate_id] < 1e-6
    for path, expected in registered['hashes'].items():
        assert digest(ROOT / path) == expected
    verification = {'status': 'passed', 'case_count': len(rows),
                    'independent_metric_max_error': maximum_metric,
                    'index_unit_reconstruction_max_error': max(unit.values()),
                    'lot_rounding_and_broker_fills_verified': False,
                    'protected_inputs_unchanged': True, 'investment_admission': False}
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'results.json', {'parent_run': registered['parent_run'],
                                            'candidates': registered['candidates'],
                                            'rows': rows, 'verification': verification,
                                            'production_changed': False,
                                            'investment_admission': False})
    write_json(directory / 'status.json', {'phase': 'complete', 'tasks': [
        {'task': 'replay', 'status': 'completed'}, {'task': 'validation', 'status': 'completed'}]})
    print(json.dumps({'directory': str(directory), 'verification': verification,
                      'rows': [{k: r[k] for k in ('id', 'cagr', 'max_drawdown',
                                                  'worst_full_year', 'dominates_red_benchmark')}
                               for r in rows]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', type=Path)
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'output/dividend_band/data_20261111')
    parser.add_argument('--run', type=Path, nargs=2)
    args = parser.parse_args()
    if args.register:
        register(args.register, args.data)
    elif args.run:
        run(args.run[0], args.run[1])
    else:
        parser.error('需要--register <主run> 或 --run <新目录> <主run>')
