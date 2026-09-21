"""按登记协议验证30只池策略，含逐日账本和独立再建仓。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from research.ex_nasdaq import explore
from research.ex_nasdaq.core import execute, SYMBOLS
from research.ex_nasdaq.explore import BASE, OUT, write_json
from research.sp500.validate import relative_metrics, ledger_audit
from src.simulation import block_bootstrap_mean


def task(job, registry, policy, directory):
    candidate, name, start = job['candidate'], job['name'], job['start']
    tested_policy = {**policy, **job.get('policy', {})}
    kwargs = job.get('stress', {})
    frame, trades, issues = execute(explore._MARKET, candidate, tested_policy, start, **kwargs)
    metrics, benchmark = relative_metrics(frame, BASE/registry['benchmark'], registry['comparison'])
    metrics.update({'issues': issues, 'trade_count': len(trades),
                    'traded_symbols': sorted(trades.symbol.unique()) if not trades.empty else [],
                    'average_exposure': float((frame.value/frame.account).mean()),
                    'ledger': ledger_audit(explore._MARKET, frame, trades, tested_policy,
                                           tested_policy['initial_capital'] * kwargs.get('capital_multiplier', 1))})
    frame.to_csv(Path(directory)/f'report_{name}.csv')
    trades.to_csv(Path(directory)/f'trades_{name}.csv', index=False)
    benchmark.to_csv(Path(directory)/f'benchmark_{name}.csv')
    return name, metrics


def validate(protocol_path, workers=3):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    exploration = BASE/protocol['exploration']
    registry = json.loads((exploration/'registry.json').read_text(encoding='utf-8'))
    manifest = json.loads((exploration/'manifest.json').read_text(encoding='utf-8'))
    # 验证用当前执行实现须与候选回放一致；新增的验证/报告工具不会改动原策略。
    mismatches = [p for p,h in manifest['hashes'].items() if hashlib.sha256((BASE/p).read_bytes()).hexdigest() != h]
    if mismatches:
        raise ValueError(f'候选冻结输入发生变化，须登记重跑：{mismatches}')
    primary = next(c for c in registry['candidates'] if c['name'] == protocol['primary'])
    policy = {**json.loads((BASE/registry['source']/'protocol.json').read_text(encoding='utf-8')),
              **registry['execution'], **primary.get('policy_overrides', {})}
    jobs = [{'name': f'base_{label}', 'candidate': primary, 'start': start, 'section': 'base'}
            for label,start in registry['starts'].items()]
    for item in protocol['stress']:
        jobs.append({'name': item['name'], 'candidate': primary, 'start': registry['starts']['long'],
                     'stress': item['kwargs'], 'section': 'stress'})
    for item in protocol['neighbors']:
        candidate = {**copy.deepcopy(primary), **item['overrides']}
        jobs.append({'name': item['name'], 'candidate': candidate, 'start': registry['starts']['long'], 'section': 'neighbors'})
    for item in protocol.get('policy_variants', []):
        jobs.append({'name': item['name'], 'candidate': primary, 'start': registry['starts']['long'],
                     'policy': item['overrides'], 'section': 'additional'})
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('验证情景名称重复')
    directory = OUT/'validation'/pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {'protocol': protocol, 'registry': registry, 'candidate': primary, 'policy': policy, 'jobs': jobs,
              'symbols': sorted(SYMBOLS), 'created_at': pd.Timestamp.now(tz='UTC').isoformat()}
    hashes = dict(manifest['hashes'])
    for p in list(Path(__file__).parent.glob('*.py'))+[protocol_path.resolve()]:
        hashes[p.relative_to(BASE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    frozen['hashes'] = hashes
    write_json(directory/'frozen_validation.json', frozen)
    for relative in hashes:
        if relative.endswith('.py'):
            target = directory/'code_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE/relative, target)
    pointer = (BASE/'output/current_run.json').read_bytes()
    results = {'base': {}, 'stress': {}, 'neighbors': {}, 'additional': {}}
    print(f'验证目录：{directory}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=explore.initialize, initargs=(BASE/registry['source'],)) as pool:
        futures = {pool.submit(task, job, registry, policy, directory): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            name, metrics = future.result()
            results[job['section']][name] = metrics
            write_json(directory/'results.json', results)
            print(f"{name}: 累计{metrics['cumulative_return']:.2%} 年化{metrics['cagr']:.2%} 回撤{metrics['max_drawdown']:.2%} 达标={metrics['target_met']}", flush=True)
    reproduction = {}
    for label in registry['starts']:
        original = pd.read_csv(exploration/f"report_{primary['name']}_{label}.csv", index_col=0, parse_dates=True)
        repeated = pd.read_csv(directory/f'report_base_{label}.csv', index_col=0, parse_dates=True)
        if not original.index.equals(repeated.index):
            raise ValueError('重放日期不同')
        reproduction[label] = float((original.account-repeated.account).abs().max())
    budget = registry['research_drawdown_budget']
    neighbors = results['neighbors']
    checks = {
        'both_base_targets_met': all(v['target_met'] for v in results['base'].values()),
        'both_base_drawdowns_within_budget': all(v['max_drawdown'] >= -budget for v in results['base'].values()),
        'stress_positive_within_budget': len(results['stress']) == 6 and all(v['cumulative_return'] > 0 and v['max_drawdown'] >= -budget for v in results['stress'].values()),
        'neighbor_count': len(neighbors),
        'neighbors_passing': sum(v['target_met'] and v['max_drawdown'] >= -budget for v in neighbors.values()),
        'reproduction_error_cny': reproduction,
        'only_30_symbols_traded': all(set(v['traded_symbols']).issubset(SYMBOLS) for section in results.values() for v in section.values()),
        'production_pointer_unchanged': pointer == (BASE/'output/current_run.json').read_bytes(),
        'original_20_percent_drawdown_passed': all(v['max_drawdown'] >= -.2 for v in results['base'].values()),
    }
    checks['neighborhood_passed'] = len(neighbors) >= 3 and checks['neighbors_passing'] >= math.ceil(len(neighbors)*2/3)
    checks['historical_acceptance'] = all(checks[k] for k in ('both_base_targets_met', 'both_base_drawdowns_within_budget',
        'stress_positive_within_budget', 'only_30_symbols_traded', 'production_pointer_unchanged', 'neighborhood_passed')) and max(reproduction.values()) < .001
    if max(reproduction.values()) > .001 or not checks['production_pointer_unchanged']:
        raise ValueError(f'重放/生产状态核验失败：{checks}')
    base = pd.read_csv(directory/'report_base_long.csv', index_col=0, parse_dates=True)
    benchmark = pd.read_csv(directory/'benchmark_base_long.csv', index_col=0, parse_dates=True)
    early, _ = relative_metrics(base.loc[:'2020-12-31'], BASE/registry['benchmark'], registry['comparison'])
    net = base['return']-base.cost
    wealth = (1+net).cumprod()
    rolling = pd.DataFrame({'strategy': wealth/wealth.shift(252)-1,
                            'spy_cny': benchmark.spy_total_cny/benchmark.spy_total_cny.shift(252)-1})
    rolling['gap'] = rolling.strategy-rolling.spy_cny
    rolling.to_csv(directory/'rolling_12month.csv')
    monthly = pd.concat([wealth.resample('ME').last().pct_change(), benchmark.spy_total_cny.resample('ME').last().pct_change()], axis=1).dropna()
    results['diagnostics'] = {'early_2016_2020': early, 'monthly_correlation': float(monthly.iloc[:,0].corr(monthly.iloc[:,1])),
        'excess_daily_block_ci': block_bootstrap_mean(net-benchmark.spy_total_cny.pct_change(fill_method=None).fillna(0)),
        'worst_12month_gap': float(rolling.gap.min()), 'best_12month_gap': float(rolling.gap.max()),
        'positive_rolling_fraction': float(rolling.strategy.dropna().gt(0).mean()), 'development_only': True}
    results['checks'] = checks
    write_json(directory/'results.json', results)
    write_json(directory/'artifact_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()})
    write_json(OUT/'validation/latest.json', {'directory': str(directory)})
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    validate(args.protocol, args.workers)
