"""登记并验证共享成交额预算对原候选及三种动态配置的影响。"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.ex_nasdaq.core import market30
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, SELECTED, comparison
from research.goal30.budget_cash_replay import simulate_budgeted_cash
from research.goal30.cash_replay import audit_cash_ledger
from research.goal30.relative_allocation import allocation_factory, verify_trace
from research.goal30.verify_dividend_notices import digest, require
from research.sp500.explore import allocator
from research.sp500.validate import relative_metrics

PROTOCOL = Path(__file__).with_name('turnover_replay_protocol.json')
OUT = BASE / 'output/goal30/turnover_replay'


def fixed_raw(candidate, eligible, exiting):
    allowed = set(eligible) - set(exiting)
    raw = {s: w for s, w in candidate['weights'].items() if s in allowed}
    for original, alternatives in candidate.get('fallbacks', {}).items():
        if original not in allowed:
            for symbol, fraction in alternatives.items():
                if symbol in allowed:
                    raw[symbol] = raw.get(symbol, 0.) + candidate['weights'][original] * fraction
    return raw


def fixed_factory(trace):
    def factory(candidate, calendar, start):
        frozen_select = allocator(candidate, calendar, start)
        def select(scores, eligible, current, holding_days, returns, policy, *, mode='rank', exiting=()):
            targets = frozen_select(scores, eligible, current, holding_days, returns, policy, mode=mode, exiting=exiting)
            trace.append({'date': str(returns.index[-1].date()),
                          'raw_weights': json.dumps(fixed_raw(candidate, eligible, exiting), sort_keys=True),
                          'target_weights': json.dumps(targets, sort_keys=True),
                          'market_eligible': json.dumps(sorted(eligible)), 'exiting': json.dumps(sorted(exiting))})
            return targets
        return select
    return factory


def verify_signals(market, registration, variant, trace):
    if variant['method'] != 'fixed':
        return verify_trace(market, registration['allocation_protocol'], variant, registration['policy'], trace)
    spec, policy = registration['protocol'], registration['policy']
    eligibility = market.eligibility.pivot(index='date', columns='symbol', values='eligible')
    days = []
    for row in trace:
        day = pd.Timestamp(row['date'])
        allowed = sorted(eligibility.columns[eligibility.loc[day].fillna(False)].tolist())
        require(allowed == json.loads(row['market_eligible']), '固定配置历史资格不一致')
        raw = fixed_raw(registration['execution_adapter_candidate'], allowed, json.loads(row['exiting']))
        require(raw == json.loads(row['raw_weights']), '固定配置/资格代理与冻结规则不符')
        targets = json.loads(row['target_weights'])
        require(set(targets).issubset(raw) and sum(targets.values()) <= policy['max_gross_weight'] + 1e-10
                and all(0 < w <= policy['max_weight'] + 1e-10 for w in targets.values()), '固定配置目标违反范围或仓位限制')
        days.append(day)
    expected = market.calendar[(market.calendar >= spec['start']) & (market.calendar <= spec['end'])][:-1]
    require(pd.DatetimeIndex(days).as_unit('ns').equals(expected.as_unit('ns')), '固定配置逐日信号不完整')
    return {'signals_verified': len(days), 'fixed_allocation_and_proxy_checked': True, 'market_eligibility_rechecked': True}


def register():
    spec = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    previous = BASE / spec['previous_dynamic_archive']
    registered = json.loads((previous / 'registration.json').read_text(encoding='utf-8'))
    completed = json.loads((previous / 'completion.json').read_text(encoding='utf-8'))
    require(digest(previous / 'registration.json') == completed['registration_sha256'], '既有动态登记改变')
    require(digest(previous / 'results.json') == completed['results_sha256'], '既有动态结果改变')
    selected, _ = load_selected(SELECTED)
    require(selected['candidate'] == registered['execution_adapter_candidate']
            and selected['policy'] == registered['policy'], '冻结候选或政策改变')
    require(spec['start'] == registered['protocol']['start'] and spec['end'] == registered['protocol']['end'], '比较期间改变')
    require(spec['variants'][1:] == registered['protocol']['variants'] and spec['scenarios'] == registered['protocol']['scenarios'],
            '本阶段不可变更动态方法、参数或压力情景')
    for name, expected in registered['input_hashes'].items():
        require(digest(BASE / name) == expected, f'冻结输入改变：{name}')
    extra = [PROTOCOL, Path(__file__), Path(__file__).with_name('budget_cash_replay.py'),
             Path(__file__).with_name('turnover_budget.py'), Path(__file__).with_name('test_turnover_budget.py'),
             previous / 'registration.json', previous / 'completion.json', previous / 'results.json', previous / 'verification.json',
             *previous.glob('*.csv')]
    inputs = {**registered['input_hashes'], **{p.relative_to(BASE).as_posix(): digest(p) for p in extra}}
    directory = OUT / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'protocol': spec,
                    'allocation_protocol': registered['protocol'], 'execution_adapter_candidate': selected['candidate'],
                    'policy': selected['policy'], 'source': selected['source'], 'benchmark': selected['benchmark'],
                    'comparison': selected['comparison'], 'actions': registered['actions'], 'input_hashes': inputs}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(OUT / 'latest_registration.json', {'directory': str(directory), 'sha256': digest(directory / 'registration.json')})
    print(json.dumps({'registered': str(directory), 'cases': 12}, ensure_ascii=False), flush=True)


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT) and directory.is_dir(), '只能执行已登记的换手预算回放')
    registration = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for name, expected in registration['input_hashes'].items():
        require(digest(BASE / name) == expected, f'登记后输入改变：{name}')
    with (directory / 'started.json').open('x', encoding='utf-8') as output:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, output)
    spec, policy = registration['protocol'], registration['policy']
    market = market30(BASE / registration['source'])
    require(str(market.calendar[-1].date()) == spec['end'], '行情截止日期改变')
    prices = market.wide('adj_close')
    results = {'cases': {}, 'investment_admission': False, 'full_goal_achieved': False}
    for variant in spec['variants']:
        for scenario in spec['scenarios']:
            name, began = variant['name'] + '__' + scenario['name'], time.monotonic()
            trace = []
            factory = (fixed_factory(trace) if variant['method'] == 'fixed' else
                       allocation_factory(prices, registration['allocation_protocol'], variant, trace))
            report, trades, events, holdings, issues = simulate_budgeted_cash(market,
                registration['execution_adapter_candidate'], policy, spec['start'], registration['actions'],
                select_factory=factory, **{k: v for k, v in scenario.items() if k != 'name'})
            require((report[['turnover', 'execution_turnover', 'budget_turnover']] <= policy['max_daily_turnover'] + 1e-10).all().all(),
                    '至少一种实际成交额口径超过全天预算')
            _, benchmark = relative_metrics(report, BASE / registration['benchmark'], registration['comparison'])
            metrics = comparison(report.account, benchmark.spy_total_cny)
            max_ok = metrics['strategy']['max_drawdown'] <= policy['max_drawdown_acceptance']
            metrics['checks'].update(max_drawdown_within_original_limit=max_ok,
                annual_goal_and_max_drawdown=metrics['checks']['annual_definition_target_met'] and max_ok,
                daily_goal_and_max_drawdown=metrics['checks']['daily_definition_target_met'] and max_ok)
            ledger = audit_cash_ledger(market, report, trades, events, policy['initial_capital'])
            signal_check = verify_signals(market, registration, variant, trace)
            annual = report.account.resample('YE').last()
            annual_returns = annual.pct_change(fill_method=None)
            annual_returns.iloc[0] = annual.iloc[0] / policy['initial_capital'] - 1
            item = {**metrics, 'variant': variant, 'scenario': scenario, 'ledger': ledger, 'signal_check': signal_check,
                    'issues': issues, 'annual_returns': {str(d.year): float(v) for d, v in annual_returns.items()},
                    'average_exposure': float((report.value / report.account).mean()),
                    'maximum_actual_turnover': float(report.execution_turnover.max()),
                    'maximum_reference_turnover': float(report.turnover.max()),
                    'maximum_budget_turnover': float(report.budget_turnover.max()),
                    'sum_daily_turnover': float(report.turnover.sum()), 'seconds': time.monotonic() - began}
            for prefix, frame in [('report', report), ('trades', trades), ('actions', events),
                                  ('holdings', holdings), ('signals', pd.DataFrame(trace)), ('benchmark', benchmark)]:
                frame.to_csv(directory / f'{prefix}_{name}.csv', index=prefix in ('report', 'benchmark'))
            item['artifact_hashes'] = {p.name: digest(p) for p in directory.glob(f'*_{name}.csv')}
            results['cases'][name] = item
            write_json_atomic(directory / 'results.json', results)
            print(json.dumps({'case': name, 'metrics': metrics['strategy'], 'max_actual_turnover': item['maximum_actual_turnover'],
                              'max_reference_turnover': item['maximum_reference_turnover'], 'checks': metrics['checks'],
                              'seconds': item['seconds']}), flush=True)
    require(len(results['cases']) == spec['evaluation']['cases'] == 12, '未完成全部12情景')
    for name, expected in registration['input_hashes'].items():
        require(digest(BASE / name) == expected, f'运行中输入改变：{name}')
    write_json_atomic(directory / 'completion.json', {'completed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'cases': len(results['cases']), 'results_sha256': digest(directory / 'results.json'),
        'registration_sha256': digest(directory / 'registration.json')})
    write_json_atomic(OUT / 'latest_completed.json', {'directory': str(directory)})
    print(json.dumps({'completed': str(directory), 'cases': 12}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run)
