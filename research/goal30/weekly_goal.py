"""周度运行与双档目标：每周首个交易日评估、触发才调仓的固定候选风险叠加研究。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.artifacts import write_json_atomic
from research.ex_nasdaq.core import market30, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30 import risk_overlay
from research.goal30.audit import BASE, SELECTED, comparison
from research.goal30.cash_replay import audit_cash_ledger, simulate_cash
from research.sp500.explore import bounded_targets
from research.sp500.validate import relative_metrics
from src.portfolio import plan_orders

PROTOCOL = Path(__file__).with_name('weekly_goal_protocol.json')
OUT = BASE / 'output/goal30/weekly_goal'
CASES = 18


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def weekly_runner(prices, variant, trace):
    """每周首个观察交易日重算门槛与目标；其余交易日不产生任何订单。"""
    gate = risk_overlay.trend_gate(prices, variant['trend_window'])
    cadence = {'week': None, 'decision': None}

    def factory(candidate, calendar, start):
        validate_candidate(candidate)
        require(candidate['family'] == 'fixed' and not candidate.get('execution_window'),
                '周度研究只支持当前冻结固定配置')

        def select(scores, eligible, current, holding_days, returns, policy, *, mode='rank', exiting=()):
            date = returns.index[-1]
            key = str(date.to_period('W-SUN'))
            if key != cadence['week']:
                require(date in gate.index, '趋势信号日缺失')
                cadence['week'] = key
                cadence['decision'] = date
                raw = risk_overlay.allocated_weights(candidate, eligible, exiting, gate.loc[date])
                targets = bounded_targets(raw, current, returns, policy)
                trace.append({'date': str(date.date()), 'week': key,
                              'raw_weights': json.dumps(raw, sort_keys=True),
                              'target_weights': json.dumps(targets, sort_keys=True),
                              'market_eligible': json.dumps(sorted(eligible)),
                              'exiting': json.dumps(sorted(exiting))})
                return targets
            return {symbol: float(weight) for symbol, weight in current.items() if weight > 1e-9}
        return select

    def plan(targets, quantities, cash, row_prices, adv, policy):
        # 只有决策日当天可以下单；其余日期的目标与持仓在份额取整下可能产生碎单，必须封闭。
        if cadence['decision'] is None or pd.Timestamp(row_prices.name) != cadence['decision']:
            return []
        return plan_orders(targets, quantities, cash, row_prices, adv, policy)
    return factory, plan


def weekly_overlay_allocator(prices, variant, trace):
    """周度决策工厂入口；与 risk_overlay.overlay_allocator 同形，供 cash_replay.allocator 打桩。

    非决策日的下单封闭函数挂在返回值的 plan 属性上：回放时必须连同
    cash_replay.plan_orders 一起打桩，否则份额取整会在周中产生碎单。
    """
    factory, plan = weekly_runner(prices, variant, trace)
    factory.plan = plan
    return factory


def decision_days(dates, start, trace):
    """决策日必须覆盖每个自然周的首个观察交易日，且一周至多一次。"""
    decisions = [pd.Timestamp(row['date']) for row in trace]
    require(decisions == sorted(set(decisions)), '决策日重复或乱序')
    require(bool(decisions), '缺少决策日')
    first_sessions = pd.Series(dates, index=dates).groupby(dates.to_period('W-SUN')).min().tolist()
    expected = {day for day in first_sessions
                if pd.Timestamp(start) <= day < dates[-1] and day >= dates[1]}
    expected.add(decisions[0])
    require(set(decisions) == expected, '决策日必须为每个自然周的首个观察交易日')
    return decisions


def cadence_audit(dates, start, trace, trades):
    decisions = decision_days(dates, start, trace)
    follow = {day: dates[dates.searchsorted(day) + 1] for day in decisions
              if dates.searchsorted(day) + 1 < len(dates)}
    require(set(follow) == set(decisions), '决策日缺少次一交易日')
    allowed = set(follow.values())
    outside = sorted({str(pd.Timestamp(value).date()) for value in pd.to_datetime(trades['date']).unique()
                      if pd.Timestamp(value) not in allowed}) if not trades.empty else []
    require(not outside, f'非决策日出现成交：{outside[:5]}')
    return {'decision_days': len(decisions), 'weeks_expected': len(set(decisions)),
            'first_decision': str(decisions[0].date()), 'last_decision': str(decisions[-1].date()),
            'trades_only_after_decisions': True, 'trades_outside': outside}


def goal_checks(metrics, goals):
    drawdown = float(metrics['max_drawdown'])
    cagr = float(metrics['cagr'])
    return {'cagr': cagr, 'max_drawdown': drawdown,
            'conservative_met': bool(cagr >= goals['conservative']['cagr']
                                     and drawdown <= goals['conservative']['max_drawdown']),
            'advanced_met': bool(cagr >= goals['advanced']['cagr']
                                 and drawdown <= goals['advanced']['max_drawdown']),
            'conservative_target': goals['conservative'], 'advanced_target': goals['advanced']}


def register():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    selected, _ = load_selected(SELECTED)
    baseline = BASE / protocol['baseline_archive']
    frozen = json.loads((baseline / 'frozen_protocol.json').read_text(encoding='utf-8'))
    require(selected['candidate'] == frozen['candidate'] and selected['policy'] == frozen['policy'],
            '基础候选或政策发生变化')
    require(protocol['start'] == frozen['protocol']['start'] and protocol['end'] == frozen['protocol']['end'],
            '周度研究必须使用相同严格十年区间')
    for path, expected in frozen['input_hashes'].items():
        require(digest(BASE / path) == expected, f'冻结输入改变：{path}')
    require(protocol['cadence'] == 'weekly_first_session', '周度节奏口径缺失')
    require(len(protocol['variants']) == len({row['name'] for row in protocol['variants']}) == 6,
            '变体缺失或重复')
    require(len(protocol['scenarios']) == len({row['name'] for row in protocol['scenarios']}) == 3,
            '压力情景缺失或重复')
    for variant in protocol['variants']:
        require(variant.get('preset') in ('conservative', 'advanced'), '档位标记无效')
        risk_overlay.risk_policy(selected['policy'], variant)
    directory = OUT / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {**frozen['input_hashes'], **{p.relative_to(BASE).as_posix(): digest(p) for p in
              [PROTOCOL, Path(__file__), SELECTED, baseline / 'frozen_protocol.json',
               baseline / 'results.json', *baseline.glob('*.csv')]}}
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                    'protocol': protocol, 'candidate': selected['candidate'], 'policy': selected['policy'],
                    'source': selected['source'], 'benchmark': selected['benchmark'],
                    'comparison': selected['comparison'], 'actions': frozen['actions'],
                    'input_hashes': inputs}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(OUT / 'latest_registration.json', {'directory': str(directory),
                      'registration_sha256': digest(directory / 'registration.json')})
    print(json.dumps({'registered': str(directory), 'cases': CASES}, ensure_ascii=False), flush=True)
    return directory


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT) and directory.is_dir(), '只能运行已登记周度研究目录')
    registration = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'登记后输入发生变化：{path}')
    with (directory / 'started.json').open('x', encoding='utf-8') as output:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, output)
    spec = registration['protocol']
    market = market30(BASE / registration['source'])
    prices = market.wide('adj_close')
    results = {'cases': {}, 'goals': spec['goals'], 'cadence': spec['cadence'],
               'investment_admission': False, 'full_goal_achieved': False,
               'research_status': spec['research_status']}
    for variant in spec['variants']:
        for scenario in spec['scenarios']:
            name = variant['name'] + '__' + scenario['name']
            began = time.monotonic()
            policy = risk_overlay.risk_policy(registration['policy'], variant)
            trace = []
            selector = weekly_overlay_allocator(prices, variant, trace)
            with patch('research.goal30.cash_replay.allocator', selector), \
                 patch('research.goal30.cash_replay.plan_orders', selector.plan):
                report, trades, actions, holdings, issues = simulate_cash(
                    market, registration['candidate'], policy, spec['start'], registration['actions'],
                    **{key: value for key, value in scenario.items() if key != 'name'})
            _, benchmark = relative_metrics(report, BASE / registration['benchmark'], registration['comparison'])
            metrics = comparison(report.account, benchmark.spy_total_cny)
            ledger = audit_cash_ledger(market, report, trades, actions, policy['initial_capital'])
            signal_check = risk_overlay.verify_trace(market, registration['candidate'], variant, trace)
            cadence = cadence_audit(market.calendar, spec['start'], trace, trades)
            annual = report.account.resample('YE').last()
            annual_returns = annual.pct_change(fill_method=None)
            annual_returns.iloc[0] = annual.iloc[0] / policy['initial_capital'] - 1
            output = {**metrics, 'goals': goal_checks(metrics['strategy'], spec['goals']),
                      'variant': variant, 'scenario': scenario, 'ledger': ledger,
                      'signal_check': signal_check, 'cadence_check': cadence, 'issues': issues,
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
            print(json.dumps({'case': name, 'metrics': metrics['strategy'], 'goals': output['goals'],
                              'seconds': output['seconds']}, ensure_ascii=True), flush=True)
    require(len(results['cases']) == CASES, '登记情景未全部完成')
    for path, expected in registration['input_hashes'].items():
        require(digest(BASE / path) == expected, f'运行中输入发生变化：{path}')
    write_json_atomic(directory / 'completion.json',
                      {'completed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'cases': CASES,
                       'results_sha256': digest(directory / 'results.json'),
                       'registration_sha256': digest(directory / 'registration.json')})
    write_json_atomic(OUT / 'latest_completed.json', {'directory': str(directory)})
    print(json.dumps({'completed': str(directory), 'cases': CASES}, ensure_ascii=False), flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run)
