"""登记及回放权益/黄金动态配置机制；不修改冻结现金引擎或生产策略。"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, market30, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, SELECTED, comparison
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.goal30.verify_dividend_notices import digest, require
from research.sp500.explore import bounded_targets
from research.sp500.validate import relative_metrics

PROTOCOL = Path(__file__).with_name('relative_allocation_protocol.json')
OUT = BASE / 'output/goal30/relative_allocation'


def validate_spec(spec):
    symbols, windows = spec['signal_symbols'], spec['lookbacks']
    require(len(symbols) == len(set(symbols)) and set(symbols).issubset(SYMBOLS), '信号标的重复或池外')
    require(len(windows) == len(set(windows)) > 0 and all(type(n) is int and n >= 2 for n in windows), '动量期限无效')
    require(type(spec['volatility_window']) is int and 2 <= spec['volatility_window'] <= max(windows), '波动窗口无效')
    require(type(spec['annualization_days']) is int and spec['annualization_days'] > 0, '年化日数无效')
    require(np.isfinite(spec['volatility_floor']) and 0 < spec['volatility_floor'] <= 1, '波动下限无效')


def feature_scores(prices, spec):
    validate_spec(spec)
    require(prices.index.is_monotonic_increasing and prices.index.is_unique, '价格日期重复或乱序')
    prices = prices.reindex(columns=spec['signal_symbols'])
    valid = np.isfinite(prices) & prices.gt(0)
    clean = prices.where(valid)
    count = max(spec['lookbacks']) + 1
    history = valid.rolling(count, min_periods=count).sum().eq(count)
    volatility = (clean.pct_change(fill_method=None).rolling(spec['volatility_window'],
                  min_periods=spec['volatility_window']).std(ddof=1) * np.sqrt(spec['annualization_days']))
    volatility = volatility.clip(lower=spec['volatility_floor'])
    return {n: ((np.log(clean / clean.shift(n)) * spec['annualization_days'] / n) / volatility)
                .where(history).replace([np.inf, -np.inf], np.nan) for n in spec['lookbacks']}


def raw_allocation(scores, method, eligible, exiting, gross):
    require(method in ('proportional', 'votes', 'equal'), '未知配置机制')
    require(np.isfinite(gross) and 0 < gross <= 1, '总预算无效')
    allowed = sorted((set(eligible) - set(exiting)) & set(scores.columns))
    usable = scores.reindex(columns=allowed).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if usable.empty:
        return {}
    if method == 'votes':
        weights = {}
        for _, row in usable.iterrows():
            best = row.max()
            if best <= 0:
                continue
            winners = row.index[row.eq(best)].tolist()
            for symbol in winners:
                weights[symbol] = weights.get(symbol, 0.) + gross / len(scores) / len(winners)
        return weights
    strength = usable.mean().clip(lower=0)
    strength = strength[strength > 0]
    if strength.empty:
        return {}
    if method == 'equal':
        return {symbol: gross / len(strength) for symbol in strength.index}
    return (strength / strength.sum() * gross).to_dict()


def allocation_factory(prices, spec, variant, trace):
    features = feature_scores(prices, spec)

    def factory(candidate, calendar, start):
        # 仅复用固定候选的现金执行接口；实际分配规则由登记的spec/variant及逐日trace证明。
        validate_candidate(candidate)

        def select(scores, eligible, current, holding_days, returns, policy, *, mode='rank', exiting=()):
            day = returns.index[-1]
            matrix = pd.DataFrame({n: frame.loc[day] for n, frame in features.items()}).T
            raw = raw_allocation(matrix, variant['method'], eligible, exiting, policy['max_gross_weight'])
            targets = bounded_targets(raw, current, returns, policy)
            trace.append({'date': str(day.date()), 'raw_weights': json.dumps(raw, sort_keys=True),
                          'target_weights': json.dumps(targets, sort_keys=True),
                          'market_eligible': json.dumps(sorted(eligible)), 'exiting': json.dumps(sorted(exiting))})
            return targets
        return select
    return factory


def prefix_scores(prices, spec):
    """用独立价格前缀直接重算末日得分，不读取预先生成的滚动特征。"""
    result = pd.DataFrame(np.nan, index=spec['lookbacks'], columns=spec['signal_symbols'])
    count = max(spec['lookbacks']) + 1
    for symbol in spec['signal_symbols']:
        values = prices[symbol].tail(count).to_numpy(dtype=float)
        if len(values) != count or not np.isfinite(values).all() or (values <= 0).any():
            continue
        daily = values[1:] / values[:-1] - 1
        vol = max(float(np.std(daily[-spec['volatility_window']:], ddof=1)) *
                  np.sqrt(spec['annualization_days']), spec['volatility_floor'])
        for n in spec['lookbacks']:
            result.at[n, symbol] = np.log(values[-1] / values[-n - 1]) * spec['annualization_days'] / n / vol
    return result


def verify_trace(market, spec, variant, policy, trace):
    prices = market.wide('adj_close')
    eligibility = market.eligibility.pivot(index='date', columns='symbol', values='eligible')
    days = []
    for row in trace:
        day = pd.Timestamp(row['date'])
        scores = prefix_scores(prices.loc[:day], spec)
        eligible = sorted(eligibility.columns[eligibility.loc[day].fillna(False)].tolist())
        require(json.loads(row['market_eligible']) == eligible, '信号资格集合与历史市场不符')
        raw = raw_allocation(scores, variant['method'], eligible, json.loads(row['exiting']), policy['max_gross_weight'])
        actual = json.loads(row['raw_weights'])
        require(set(raw) == set(actual) and all(abs(raw[s] - actual[s]) < 1e-10 for s in raw), '信号与当日价格前缀不符')
        targets = json.loads(row['target_weights'])
        require(set(targets).issubset(raw) and sum(targets.values()) <= policy['max_gross_weight'] + 1e-10,
                '目标出现信号之外资产或超总仓位')
        require(all(0 < w <= policy['max_weight'] + 1e-10 for w in targets.values()), '目标单只权重无效')
        days.append(day)
    expected_days = market.calendar[(market.calendar >= spec['start']) & (market.calendar <= spec['end'])][:-1]
    require(pd.DatetimeIndex(days).as_unit('ns').equals(expected_days.as_unit('ns')), '逐日信号日期不完整或重复')
    return {'signals_verified': len(days), 'independent_price_prefix_check': True, 'market_eligibility_rechecked': True}


def register():
    spec = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    validate_spec(spec)
    selected, _ = load_selected(SELECTED)
    baseline = BASE / spec['baseline_archive']
    frozen = json.loads((baseline / 'frozen_protocol.json').read_text(encoding='utf-8'))
    require(selected['candidate'] == frozen['candidate'] and selected['policy'] == frozen['policy'], '基础候选或政策改变')
    require(spec['start'] == frozen['protocol']['start'] and spec['end'] == frozen['protocol']['end'], '十年区间改变')
    require(set(spec['signal_symbols']) == {'SH510300', 'SH515880', 'SH518880'}, '登记的原三只资产范围改变')
    require(len(spec['variants']) == len({v['name'] for v in spec['variants']}) == 3, '方法重复或缺漏')
    require(len(spec['scenarios']) == len({v['name'] for v in spec['scenarios']}) == 3, '压力情景重复或缺漏')
    for path, expected in frozen['input_hashes'].items():
        require(digest(BASE / path) == expected, f'冻结输入改变：{path}')
    extra = [PROTOCOL, Path(__file__), Path(__file__).with_name('test_relative_allocation.py'), SELECTED,
             baseline / 'frozen_protocol.json', baseline / 'results.json', *baseline.glob('*.csv'),
             BASE / 'output/goal30/diagnostics/drawdown_duration/results.json',
             BASE / 'output/goal30/verification/legacy_primary/verified_primary.json']
    inputs = {**frozen['input_hashes'], **{p.relative_to(BASE).as_posix(): digest(p) for p in extra}}
    directory = OUT / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'protocol': spec,
                    'execution_adapter_candidate': selected['candidate'], 'policy': selected['policy'],
                    'source': selected['source'], 'benchmark': selected['benchmark'], 'comparison': selected['comparison'],
                    'actions': frozen['actions'], 'input_hashes': inputs}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(OUT / 'latest_registration.json', {'directory': str(directory), 'sha256': digest(directory / 'registration.json')})
    print(json.dumps({'registered': str(directory), 'cases': 9}, ensure_ascii=False), flush=True)


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT) and directory.is_dir(), '只能运行已登记动态配置目录')
    registration = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'登记后输入改变：{path}')
    with (directory / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, stream)
    spec, policy = registration['protocol'], registration['policy']
    market = market30(BASE / registration['source'])
    require(str(market.calendar[-1].date()) == spec['end'], '行情截止日与登记不符')
    prices = market.wide('adj_close')
    results = {'cases': {}, 'research_status': spec['research_status'], 'investment_admission': False, 'full_goal_achieved': False}
    for variant in spec['variants']:
        for scenario in spec['scenarios']:
            name, began = variant['name'] + '__' + scenario['name'], time.monotonic()
            trace = []
            with patch('research.goal30.cash_replay.allocator', allocation_factory(prices, spec, variant, trace)):
                report, trades, events, holdings, issues = simulate_cash(market,
                    registration['execution_adapter_candidate'], policy, spec['start'], registration['actions'],
                    **{key: value for key, value in scenario.items() if key != 'name'})
            _, benchmark = relative_metrics(report, BASE / registration['benchmark'], registration['comparison'])
            metrics = comparison(report.account, benchmark.spy_total_cny)
            max_ok = metrics['strategy']['max_drawdown'] <= policy['max_drawdown_acceptance']
            metrics['checks'].update(max_drawdown_within_original_limit=max_ok,
                annual_goal_and_max_drawdown=metrics['checks']['annual_definition_target_met'] and max_ok,
                daily_goal_and_max_drawdown=metrics['checks']['daily_definition_target_met'] and max_ok)
            ledger = audit_cash_ledger(market, report, trades, events, policy['initial_capital'])
            signal_check = verify_trace(market, spec, variant, policy, trace)
            annual = report.account.resample('YE').last()
            annual_returns = annual.pct_change(fill_method=None)
            annual_returns.iloc[0] = annual.iloc[0] / policy['initial_capital'] - 1
            item = {**metrics, 'variant': variant, 'scenario': scenario, 'ledger': ledger, 'signal_check': signal_check,
                    'issues': issues, 'annual_returns': {str(d.year): float(v) for d, v in annual_returns.items()},
                    'average_exposure': float((report.value / report.account).mean()),
                    'sum_daily_turnover': float(report.turnover.sum()), 'seconds': time.monotonic() - began}
            for prefix, frame in [('report', report), ('trades', trades), ('actions', events),
                                  ('holdings', holdings), ('signals', pd.DataFrame(trace)), ('benchmark', benchmark)]:
                frame.to_csv(directory / f'{prefix}_{name}.csv', index=prefix in ('report', 'benchmark'))
            item['artifact_hashes'] = {p.name: digest(p) for p in directory.glob(f'*_{name}.csv')}
            results['cases'][name] = item
            write_json_atomic(directory / 'results.json', results)
            print(json.dumps({'case': name, 'metrics': metrics['strategy'], 'checks': metrics['checks'], 'seconds': item['seconds']}), flush=True)
    require(len(results['cases']) == 9, '未完成全部9情景')
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'运行中输入改变：{path}')
    write_json_atomic(directory / 'completion.json', {'completed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'cases': len(results['cases']), 'results_sha256': digest(directory / 'results.json'),
        'registration_sha256': digest(directory / 'registration.json')})
    write_json_atomic(OUT / 'latest_completed.json', {'directory': str(directory)})
    print(json.dumps({'completed': str(directory), 'cases': 9}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run)
