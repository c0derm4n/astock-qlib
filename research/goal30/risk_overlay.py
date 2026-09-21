"""登记并回放固定30只池候选的风险方案；只新增研究，不修改冻结引擎。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import market30, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, SELECTED, comparison
from research.goal30.cash_replay import simulate_cash, audit_cash_ledger
from research.sp500.explore import bounded_targets
from research.sp500.validate import relative_metrics

PROTOCOL = Path(__file__).with_name('risk_overlay_protocol.json')
OUT = BASE / 'output/goal30/risk_overlay'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def risk_policy(policy, variant):
    result = copy.deepcopy(policy)
    window, target = variant['trend_window'], variant['vol_target']
    require(window is None or (type(window) is int and window >= 2), '趋势窗口须为至少2日的整数')
    require(target is None or (not isinstance(target, bool) and np.isfinite(target) and 0 < target <= 1),
            '波动目标须位于0至1之间')
    if target is not None:
        result.update(volatility_policy='target', vol_target=target)
    return result


def trend_gate(prices, window):
    require(prices.index.is_monotonic_increasing and prices.index.is_unique, '价格日期乱序或重复')
    if window is None:
        return np.isfinite(prices) & prices.gt(0)
    require(type(window) is int and window >= 2, '趋势窗口无效')
    usable = prices.where(np.isfinite(prices) & prices.gt(0))
    return usable.gt(usable.rolling(window, min_periods=window).mean())


def allocated_weights(candidate, eligible, exiting, gate):
    eligible, exiting = set(eligible), set(exiting)
    weights = {}
    for original, weight in candidate['weights'].items():
        alternatives = ({original: 1.} if original in eligible
                        else candidate.get('fallbacks', {}).get(original, {}))
        for symbol, fraction in alternatives.items():
            if symbol in eligible and symbol not in exiting and bool(gate.get(symbol, False)):
                weights[symbol] = weights.get(symbol, 0.) + weight * fraction
    return weights


def overlay_allocator(prices, variant, trace):
    gate = trend_gate(prices, variant['trend_window'])

    def factory(candidate, calendar, start):
        validate_candidate(candidate)
        require(candidate['family'] == 'fixed' and not candidate.get('execution_window'),
                '风险回放只支持当前每日固定配置')

        def select(scores, eligible, current, holding_days, returns, policy, *, mode='rank', exiting=()):
            date = returns.index[-1]
            require(date in gate.index, '趋势信号日缺失')
            raw = allocated_weights(candidate, eligible, exiting, gate.loc[date])
            targets = bounded_targets(raw, current, returns, policy)
            trace.append({'date': str(date.date()), 'raw_weights': json.dumps(raw, sort_keys=True),
                          'target_weights': json.dumps(targets, sort_keys=True),
                          'market_eligible': json.dumps(sorted(eligible)), 'exiting': json.dumps(sorted(exiting))})
            return targets
        return select
    return factory


def verify_trace(market, candidate, variant, trace):
    # 从每个信号日的原价格前缀重算均价，避免用已生成gate充当自身正确性证据。
    prices = market.wide('adj_close')
    window = variant['trend_window']
    rows = []
    for row in trace:
        date = pd.Timestamp(row['date'])
        required = set(candidate['weights']) | {s for alt in candidate.get('fallbacks', {}).values() for s in alt}
        gate = {}
        for symbol in required:
            history = prices.loc[:date, symbol]
            if window is None:
                gate[symbol] = bool(np.isfinite(history.iloc[-1]) and history.iloc[-1] > 0)
            else:
                recent = history.tail(window)
                gate[symbol] = bool(len(recent) == window and np.isfinite(recent).all()
                                    and recent.gt(0).all() and recent.iloc[-1] > recent.mean())
        expected = allocated_weights(candidate, json.loads(row['market_eligible']), json.loads(row['exiting']), gate)
        actual = json.loads(row['raw_weights'])
        require(expected == actual, f'信号日{date}的趋势配置不符合历史前缀规则')
        targets = json.loads(row['target_weights'])
        require(set(targets).issubset(actual), '风险约束生成了趋势门槛之外的持仓')
        rows.append(date)
    require(len(rows) == len(set(rows)), '重复信号日')
    return {'signals_verified': len(rows), 'independent_prefix_trend_check': True}


def register():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    selected, _ = load_selected(SELECTED)
    baseline = BASE / protocol['baseline_archive']
    frozen = json.loads((baseline / 'frozen_protocol.json').read_text(encoding='utf-8'))
    require(selected['candidate'] == frozen['candidate'] and selected['policy'] == frozen['policy'],
            '基础候选或政策发生变化')
    require(protocol['start'] == frozen['protocol']['start'] and protocol['end'] == frozen['protocol']['end'],
            '新方案必须使用相同严格十年区间')
    for path, expected in frozen['input_hashes'].items():
        require(digest(BASE / path) == expected, f'冻结输入改变：{path}')
    require(len(protocol['variants']) == len({r['name'] for r in protocol['variants']}) == 6, '方案缺失或重复')
    require(len(protocol['scenarios']) == len({r['name'] for r in protocol['scenarios']}) == 3, '压力情景缺失或重复')
    for variant in protocol['variants']:
        risk_policy(selected['policy'], variant)
    directory = OUT / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {**frozen['input_hashes'], **{p.relative_to(BASE).as_posix(): digest(p) for p in
              [PROTOCOL, Path(__file__), SELECTED, baseline / 'frozen_protocol.json', baseline / 'results.json',
               *baseline.glob('*.csv')]}}
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                    'protocol': protocol, 'candidate': selected['candidate'], 'policy': selected['policy'],
                    'source': selected['source'], 'benchmark': selected['benchmark'],
                    'comparison': selected['comparison'], 'actions': frozen['actions'], 'input_hashes': inputs}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(OUT / 'latest_registration.json', {'directory': str(directory),
                      'registration_sha256': digest(directory / 'registration.json')})
    print(json.dumps({'registered': str(directory), 'cases': 18}, ensure_ascii=False), flush=True)
    return directory


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT) and directory.is_dir(), '只能运行已登记风险研究目录')
    registration = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'登记后输入发生变化：{path}')
    # 排他启动；中断时保留已完成情景，避免把旧状态文件当作仍在运行的进程。
    with (directory / 'started.json').open('x', encoding='utf-8') as output:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, output)
    spec = registration['protocol']
    market = market30(BASE / registration['source'])
    prices = market.wide('adj_close')
    results = {'cases': {}, 'investment_admission': False, 'full_goal_achieved': False,
               'research_status': spec['research_status']}
    for variant in spec['variants']:
        for scenario in spec['scenarios']:
            name = variant['name'] + '__' + scenario['name']
            began = time.monotonic()
            policy = risk_policy(registration['policy'], variant)
            trace = []
            with patch('research.goal30.cash_replay.allocator', overlay_allocator(prices, variant, trace)):
                report, trades, actions, holdings, issues = simulate_cash(market, registration['candidate'], policy,
                    spec['start'], registration['actions'],
                    **{key: value for key, value in scenario.items() if key != 'name'})
            _, benchmark = relative_metrics(report, BASE / registration['benchmark'], registration['comparison'])
            metrics = comparison(report.account, benchmark.spy_total_cny)
            maximum_passes = metrics['strategy']['max_drawdown'] <= registration['policy']['max_drawdown_acceptance']
            metrics['checks'].update(max_drawdown_within_original_limit=maximum_passes,
                annual_goal_and_max_drawdown=metrics['checks']['annual_definition_target_met'] and maximum_passes,
                daily_goal_and_max_drawdown=metrics['checks']['daily_definition_target_met'] and maximum_passes)
            ledger = audit_cash_ledger(market, report, trades, actions, policy['initial_capital'])
            signal_check = verify_trace(market, registration['candidate'], variant, trace)
            annual = report.account.resample('YE').last()
            annual_returns = annual.pct_change(fill_method=None)
            annual_returns.iloc[0] = annual.iloc[0] / policy['initial_capital'] - 1
            output = {**metrics, 'variant': variant, 'scenario': scenario, 'ledger': ledger,
                      'signal_check': signal_check, 'issues': issues,
                      'annual_returns': {str(k.year): float(v) for k, v in annual_returns.items()},
                      'average_exposure': float((report.value / report.account).mean()),
                      'sum_daily_turnover': float(report.turnover.sum()),
                      'seconds': time.monotonic() - began}
            for prefix, frame in [('report', report), ('trades', trades), ('actions', actions),
                                  ('holdings', holdings), ('signals', pd.DataFrame(trace)), ('benchmark', benchmark)]:
                frame.to_csv(directory / f'{prefix}_{name}.csv', index=prefix in ('report', 'benchmark'))
            output['artifact_hashes'] = {p.name: digest(p) for p in directory.glob(f'*_{name}.csv')}
            results['cases'][name] = output
            write_json_atomic(directory / 'results.json', results)
            print(json.dumps({'case': name, 'metrics': metrics['strategy'], 'checks': metrics['checks'],
                              'seconds': output['seconds']}, ensure_ascii=True), flush=True)
    require(len(results['cases']) == 18, '登记情景未全部完成')
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'运行中输入发生变化：{path}')
    write_json_atomic(directory / 'completion.json', {'completed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                      'cases': 18, 'results_sha256': digest(directory / 'results.json'),
                      'registration_sha256': digest(directory / 'registration.json')})
    write_json_atomic(OUT / 'latest_completed.json', {'directory': str(directory)})
    print(json.dumps({'completed': str(directory), 'cases': 18}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run)
