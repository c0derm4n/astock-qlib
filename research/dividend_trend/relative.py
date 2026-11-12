"""首批之后独立登记的相对强弱研究，不修改首批运行或正式策略。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from .run import (ROOT, OLD, digest, write_json, todo_write, prepare_data, protected_files,
                  replay, metrics, save_case, verify_cases)
PROTOCOL = Path(__file__).with_name('relative_protocol.json')


def candidates():
    rows = [dict(id=f'dual_mom{lookback}_ma200_v15', family='dual', lookback=lookback)
            for lookback in (63, 126, 252)]
    rows.append(dict(id='dual_blend_ma200_v15', family='dual', lookback='blend'))
    rows.extend(dict(id=f'core50_mom{lookback}_ma200_v15', family='core50', lookback=lookback)
                for lookback in (63, 126, 252))
    rows.extend([dict(id='dual_mom126_no_ma_v15', family='dual', lookback=126, no_ma=True),
                 dict(id='tilt_mom126_ma200_v15', family='tilt', lookback=126)])
    return rows


def signal_targets(prices, candidate):
    lookback = candidate['lookback']
    if lookback == 'blend':
        score = sum(prices / prices.shift(n) - 1 for n in (63, 126, 252)) / 3
    else:
        score = prices / prices.shift(lookback) - 1
    eligible = score.gt(0) & score.notna()
    if not candidate.get('no_ma'):
        eligible &= prices.gt(prices.rolling(200, min_periods=200).mean())
    ranking = score.where(eligible, -np.inf)
    red_wins = eligible.red & ranking.red.ge(ranking.gold)
    gold_wins = eligible.gold & ranking.gold.gt(ranking.red)
    target = pd.DataFrame(0., index=prices.index, columns=prices.columns)
    if candidate['family'] == 'core50':
        target['red'] = .5 + .45 * red_wins
        target['gold'] = .45 * gold_wins
    elif candidate['family'] == 'dual':
        target['red'] = .95 * red_wins
        target['gold'] = .95 * gold_wins
    elif candidate['family'] == 'tilt':
        target['red'] = .6 * red_wins + .35 * (gold_wins & eligible.red)
        target['gold'] = .6 * gold_wins + .35 * (red_wins & eligible.gold)
    else:
        raise ValueError(candidate['family'])
    returns = prices.pct_change(fill_method=None)
    variance = pd.Series(0., index=prices.index)
    for a in prices:
        for b in prices:
            variance += target[a] * target[b] * returns[a].rolling(60).cov(returns[b]) * 252
    risk = variance.clip(lower=0).pow(.5)
    scale = (.15 / risk.replace(0, np.nan)).clip(upper=1).where(risk.ne(0), 1.).fillna(0)
    target = target.mul(scale, axis=0)
    assert target.ge(0).all().all() and target.sum(axis=1).le(.95000000001).all()
    return target


def register():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    stamp = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    directory = ROOT / 'output/dividend_trend/runs' / f'{stamp}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True, exist_ok=False)
    files = protected_files() + [PROTOCOL, Path(__file__), ROOT / protocol['parent_run'] / 'results.json']
    write_json(directory / 'registration.json', {
        'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'protocol': protocol, 'candidates': candidates(),
        'hashes': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'investment_admission': False, 'production_changed': False})
    todo_write(directory, 'data')
    (directory / 'raw').mkdir()
    for path in (ROOT / protocol['parent_run'] / 'raw').glob('*'):
        shutil.copyfile(path, directory / 'raw' / path.name)
    print(str(directory))


def run(directory):
    directory = Path(directory).resolve()
    assert not (directory / 'results.json').exists(), '已完成run禁止覆盖'
    registered = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registered['hashes'].items():
        assert digest(ROOT / path) == expected, path
    protocol = registered['protocol']
    prices, etf_prices = prepare_data(directory, protocol)
    start = protocol['start']
    red_nav = 100000 * prices.red.loc[start:] / prices.red.loc[start:].iloc[0]
    benchmark = metrics(red_nav)
    red_nav.to_csv(directory / 'red_benchmark_nav.csv', index_label='date')
    passive = metrics(100000 * etf_prices.red / etf_prices.red.iloc[0])
    todo_write(directory, 'replay')
    rows, paths = [], {}
    scenarios = {'cost2': {'cost': .002}, 'delay2': {'delay': 2},
                 'fee1pct': {'haircut': .01}, 'week_end': {'week_end': True}}
    for candidate in registered['candidates']:
        sid = candidate['id']
        targets = signal_targets(prices, candidate)
        daily, trades = replay(prices, targets, start)
        rows.append(save_case(directory, sid, daily, trades, benchmark))
        paths[sid] = daily.nav
        if sid not in protocol['anchors']:
            continue
        for name, arguments in scenarios.items():
            d, t = replay(prices, targets, start, **arguments)
            rows.append(save_case(directory, f'{sid}__{name}', d, t, benchmark))
        for name, arguments in {'base': {}, 'cost2': {'cost': .002}, 'delay2': {'delay': 2}}.items():
            d, t = replay(etf_prices, targets, '2019-01-18', haircut=0,
                          no_fill_dates=['2021-10-22'], **arguments)
            rows.append(save_case(directory, f'{sid}__etf_{name}', d, t, passive))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('annual', 'personal_goals')}
                  for r in rows]).to_csv(directory / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'id': r['id'], **a} for r in rows for a in r['annual']]).to_csv(
        directory / 'annual_returns.csv', index=False, encoding='utf-8-sig')
    segments = []
    for sid, nav in {**paths, 'red_benchmark': red_nav}.items():
        for begin, end in [('2017-01-01', '2021-12-31'), ('2022-01-01', '2025-12-31'),
                           ('2017-01-01', '2024-12-31')]:
            before = nav.index[nav.index < begin][-1]
            segments.append({'id': sid, 'window': begin[:4]+'-'+end[:4], **metrics(nav.loc[before:end])})
    pd.DataFrame([{k: v for k, v in r.items() if k != 'annual'} for r in segments]).to_csv(
        directory / 'segments.csv', index=False, encoding='utf-8-sig')
    todo_write(directory, 'validation')
    verification = verify_cases(directory, prices, {**registered, 'candidates': []}, rows)
    errors = {}
    for candidate in registered['candidates']:
        prefix = prices.loc[:'2021-12-31']
        target = signal_targets(prefix, candidate)
        pd.testing.assert_frame_equal(target, signal_targets(prices, candidate).loc[prefix.index])
        daily, _ = replay(prefix, target, start)
        original = paths[candidate['id']].loc[daily.index]
        error = float((daily.nav - original).abs().max())
        assert error < 1e-7
        errors[candidate['id']] = error
    verification['causal_prefix_max_errors'] = errors
    write_json(directory / 'verification.json', verification)
    write_json(directory / 'results.json', {'protocol': protocol['version'], 'benchmark': benchmark,
        'etf_benchmark': passive, 'rows': rows, 'segments': segments,
        'investment_admission': False, 'production_changed': False})
    todo_write(directory, 'delivery')
    print(json.dumps({'directory': str(directory),
        'main_rows': [{k: v for k, v in r.items() if k != 'annual'} for r in rows if '__' not in r['id']],
        'verification': verification}, ensure_ascii=False, indent=2))


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
        parser.error('需要--register或--run')
