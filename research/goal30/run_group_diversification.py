"""登记、运行并从归档独立复核分组配置的全部15个完整十年情景。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE, comparison
from research.goal30.bond_dividend_evidence import digest
from research.goal30.group_diversification import (GROUPS, RULES, VARIANTS, decision, execution_policy,
    feature_matrix, prefix_scores)
from research.goal30.primary_cash_ledger import EventCalendar, freeze_next, new_state, settle_day, validate_state
from research.goal30.primary_research_series import ROOT, START, END, corrected_market, read
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import PrimaryMarketDays, SCENARIOS, REFERENCE, inspect_case

OUT = BASE / 'output/goal30/group_diversification'
CODE = ('group_diversification.py', 'test_group_diversification.py', 'run_group_diversification.py')


def register():
    primary = BASE / read(ROOT / 'latest_validation.json')['directory']
    proof = read(primary / 'verification.json')
    old = read(primary / 'registration.json')
    require(proof['cases_verified'] == 204 and proof['nonzero_cash_events'] == 53 and proof['nonzero_split_events'] == 14,
            '原件执行扩展尚未完整通过')
    inputs = {**old['input_hashes'], **proof['artifact_hashes']}
    files = [primary / f for f in ('registration.json', 'results.json', 'completion.json', 'verification.json')]
    files += [Path(__file__).with_name(name) for name in CODE]
    inputs.update({p.relative_to(BASE).as_posix(): digest(p) for p in files})
    for path, expected in inputs.items():
        require(digest(BASE / path) == expected, f'登记输入变化：{path}')
    directory = OUT / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'registered_code').mkdir()
    for name in CODE:
        shutil.copyfile(Path(__file__).with_name(name), directory / 'registered_code' / name)
    protocol = {'start': START, 'end': END, 'primary': VARIANTS[0]['name'], 'groups': GROUPS,
        'rules': RULES, 'variants': VARIANTS, 'scenarios': SCENARIOS, 'case_count': 15,
        'hypothesis': 'Predefined industry representatives and covariance risk budgets may reduce correlated repeated exposures and long underwater periods while retaining positive trends.',
        'research_status': 'Previously examined history; 95 legacy candidates and later failures known. New registration does not create out-of-sample evidence. No parameter selection on this batch.',
        'signal': 'Mean 63/126/252 annualized log momentum divided by 63-day annualized standard deviation floored at 5%. Require 253 consecutive actual positive closes, mean positive and at least two positive horizons. No fill of signal prices.',
        'selection': 'At first observed session of each month, select top group representatives or ungrouped controls. Group winner ties excluded. Boundary ties that do not fit are excluded together, leaving slots in cash. All 30 are candidates, subject to historical eligibility.',
        'sizing': 'Equal weights or cyclic coordinate descent equal risk contribution on 63-day covariance, 25% diagonal shrinkage. Gross allocation is 95% times filled/total slots. Cap each target at35%, original risk groups at70%, portfolio forecast volatility at15%; clipped capital remains cash.',
        'daily': 'Check once at close. Month-internal negative trend, missing history, eligibility or queued exit removes raw target until next monthly review. Reapply covariance cap and 3pp drift band; constrain again after retaining current weights. Next-session open execution,30% shared buy/sell budget.',
        'cash': 'Unchanged primary-action historical payment model; unobserved future payments remain receivable. Cash earns zero, no leverage, no synthetic index substitutions.',
        'gate': 'CAGR>=same-China-session SPY total CNY; daily mean drawdown<=SPY; retain20% MDD. Primary and >=2/3 main+3/5-slot base neighborhood must jointly pass; primary all stress must also jointly pass before further admission review.',
        'promotion': 'No change to live pointer or daily automation. Full directory, account reconciliation and126 real-forward sessions still required.'}
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'protocol': protocol,
        'policy': execution_policy(old['policy']), 'actions': old['actions'], 'input_hashes': inputs,
        'primary_execution_archive': primary.relative_to(BASE).as_posix()}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(OUT / 'latest_registration.json', {'directory': directory.relative_to(BASE).as_posix()})
    print(json.dumps({'registered': directory.relative_to(BASE).as_posix(), 'cases': 15, 'inputs': len(inputs)}), flush=True)


def feature_bank(frames):
    # 只读实际信号价格；估值允许的5日沿用不进入本策略的动量/协方差。
    prices = frames.market.wide('adj_close')
    features = feature_matrix(prices)
    dates = frames.days[(frames.days >= START) & (frames.days <= END)][:-1]
    bank, checked, error = {}, 0, 0.
    for day in dates:
        expected = prefix_scores(prices.loc[:day])
        actual = features.loc[day].dropna().to_dict()
        require(set(expected) == set(actual), '有效信号与独立价格前缀不同')
        for symbol in expected:
            delta = abs(actual[symbol] - expected[symbol])
            error = max(error, delta)
            require(delta < 1e-10, '动量分数与独立前缀不同')
        bank[str(day.date())] = expected
        checked += len(expected)
    return prices, features, bank, {'sessions': len(dates), 'valid_symbol_scores': checked,
                                  'maximum_score_error': error, 'all_thirty_checked_each_session': True}


def drive(frames, prices, features, registered, variant, scenario):
    dates = frames.days[(frames.days >= START) & (frames.days <= END)]
    policy = registered['policy']
    state = new_state(policy['initial_capital'], frames.context(dates[0])['previous_date'])
    calendar = EventCalendar(registered['actions'], frames.days, payment_mode='announced_schedule_model',
        payment_delay_sessions=scenario['payment_delay_sessions'],
        uncertain_payment_extra_sessions=scenario['uncertain_payment_extra_sessions'])
    selection = {}
    data = {'report': [], 'trades': [], 'actions': [], 'holdings': [], 'signals': [], 'state_hashes': [],
            'pending': {}, 'issue_totals': {}}
    for i, day in enumerate(dates):
        context = frames.context(day)
        data['pending'][context['date']] = state['pending_orders']
        state, output = settle_day(state, context, policy, calendar, slippage_multiplier=scenario['slippage_multiplier'])
        data['report'].append(output['report'])
        for kind in ('trades', 'actions', 'holdings'):
            data[kind].extend(output[kind])
        for key, value in output['issues'].items():
            data['issue_totals'][key] = data['issue_totals'].get(key, 0) + value
        if i+1 < len(dates):
            allowed = frames.eligible.columns[frames.eligible.loc[day].fillna(False)].tolist()
            current = {h['symbol']: h['value'] / (output['report']['cash'] + output['report']['value']) for h in output['holdings']}
            selection, targets, trace = decision(selection, features.loc[day].dropna().to_dict(), prices.loc[:day],
                allowed, state['exiting'], current, policy, variant)
            data['signals'].append({'date': context['date'], 'eligible': sorted(allowed), 'exiting': state['exiting'],
                'current_weights': current, 'selection_state': selection, 'target_weights': targets, **trace})
            state = freeze_next(state, targets, context['close'], frames.adv.loc[day].to_dict(), policy, dates[i+1])
        state = json.loads(json.dumps(state, allow_nan=False))
        validate_state(state)
        data['state_hashes'].append({'date': context['date'], 'state_sha256': state['state_sha256']})
    data['final_state'] = state
    return data


def near_mapping(actual, expected, label):
    require(set(actual) == set(expected) and all(abs(actual[s] - expected[s]) < 1e-9 for s in expected), label)


def verify_signals(frames, prices, bank, registered, variant, data):
    reports = {r['date']: r for r in data['report']}
    holdings, trades = {}, {}
    for row in data['holdings']:
        holdings.setdefault(row['date'], []).append(row)
    for row in data['trades']:
        trades.setdefault(row['date'], []).append(row)
    require([r['date'] for r in data['signals']] == list(reports)[:-1] == list(bank), '信号日缺失或重复')
    state, exiting, reviews = {}, set(), 0
    for row in data['signals']:
        day = row['date']; h = holdings.get(day, [])
        actual_qty = {v['symbol']: v['quantity'] for v in h}
        orders = {(o['symbol'], o['side']): o for o in data['pending'][day]}
        for trade in trades.get(day, []):
            if trade['side'] == 'sell' and orders[(trade['symbol'], 'sell')]['target_weight'] <= 1e-9:
                exiting.add(trade['symbol'])
        exiting.intersection_update(actual_qty)
        require(sorted(exiting) == row['exiting'], '退出状态与成交和冻结目标不符')
        report = reports[day]
        current = {v['symbol']: v['value'] / (report['cash'] + report['value']) for v in h}
        near_mapping(row['current_weights'], current, '信号持仓权重不能由现金和市值重建')
        allowed = sorted(frames.eligible.columns[frames.eligible.loc[pd.Timestamp(day)].fillna(False)].tolist())
        require(allowed == row['eligible'], '信号资格与当时行情不同')
        state, targets, trace = decision(state, bank[day], prices.loc[:day], allowed, exiting, current, registered['policy'], variant)
        require(row['selection_state']['month'] == state['month'] and row['monthly_review'] == trace['monthly_review'], '月度复查状态不同')
        near_mapping(row['selection_state']['raw_weights'], state['raw_weights'], '月内持久目标不符')
        near_mapping(row['target_weights'], targets, '目标无法从当时前缀和账户重建')
        near_mapping(row['unconstrained_risk_weights'], trace['unconstrained_risk_weights'], '月度初始风险权重不符')
        reviews += trace['monthly_review']
        require(set(targets).issubset(allowed) and not set(targets).intersection(exiting), '不合资格或退出资产仍获目标')
    # 成交资格以已冻结信号日为准；行情限制另由执行层和独立账本验证。
    for trade in data['trades']:
        if trade['side'] == 'buy':
            require(bool(frames.eligible.at[pd.Timestamp(trade['signal_date']), trade['symbol']]), '买入缺少信号日资格')
        require(trade['participation'] <= registered['policy']['max_adv_participation']+1e-12, '成交超过ADV参与率')
    return {'signals_verified': len(data['signals']), 'monthly_reviews': reviews,
            'independent_prefix_and_account_inputs': True, 'exit_state_from_fills': True}


def metrics_for(data, benchmark):
    report = pd.DataFrame(data['report']).set_index('date')
    report.index = pd.to_datetime(report.index)
    result = comparison(report.account, benchmark.spy_total_cny)
    result['checks']['maximum_drawdown_within_20pct'] = result['strategy']['max_drawdown'] <= .20
    result['checks']['joint_admission_performance'] = result['checks']['daily_definition_target_met'] and result['checks']['maximum_drawdown_within_20pct']
    annual = report.account.resample('YE').last()
    returns = annual.pct_change(fill_method=None)
    returns.iloc[0] = annual.iloc[0] / report.account.iloc[0] - 1
    bannual = benchmark.spy_total_cny.resample('YE').last().pct_change(fill_method=None)
    bannual.iloc[0] = benchmark.spy_total_cny.resample('YE').last().iloc[0] / benchmark.spy_total_cny.iloc[0] - 1
    annual_rows = {str(d.year): {'strategy': float(v), 'spy_total_cny': float(bannual.loc[d]),
        'partial_year': d.year in (2016, 2026)} for d, v in returns.items()}
    pre2025 = comparison(report.account.loc[:'2024-12-31'], benchmark.spy_total_cny.loc[:'2024-12-31'])
    return {'metrics': result, 'annual_returns': annual_rows, 'pre_2025_diagnostic': pre2025,
        'sum_daily_two_way_turnover': float(report.turnover.sum()),
        'mean_invested_fraction': float((report.value / report.account).mean()), 'trade_count': len(data['trades'])}


def check_inputs(registered):
    for path, expected in registered['input_hashes'].items():
        require(digest(BASE / path) == expected, f'登记输入改变：{path}')


def decision_summary(cases):
    primary = 'group4_risk_parity'
    primary_ok = all(cases[primary+'__'+s['name']]['metrics']['checks']['joint_admission_performance'] for s in SCENARIOS)
    neighborhood = [cases[f'group{k}_risk_parity__base']['metrics']['checks']['joint_admission_performance'] for k in (3,4,5)]
    return {'primary_all_scenarios_jointly_pass': primary_ok, 'neighborhood_jointly_pass': sum(neighborhood),
        'neighborhood_count': 3, 'eligible_for_further_admission_review': primary_ok and sum(neighborhood) >= 2,
        'investment_admission': False, 'full_goal_achieved': False, 'qualified_real_forward_sessions': 0}


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT), '须使用登记目录')
    registered = read(directory / 'registration.json')
    check_inputs(registered)
    with (directory / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, stream)
    result = {'cases': {}, 'investment_admission': False, 'full_goal_achieved': False}
    try:
        frames = PrimaryMarketDays(corrected_market())
        prices, features, bank, feature_proof = feature_bank(frames)
        result['feature_verification'] = feature_proof
        benchmark = pd.read_csv(REFERENCE / 'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True)
        for variant in registered['protocol']['variants']:
            for scenario in registered['protocol']['scenarios']:
                key = variant['name']+'__'+scenario['name']; began = time.monotonic()
                data = drive(frames, prices, features, registered, variant, scenario)
                ledger = inspect_case(frames, registered, scenario, registered['actions'], data)
                signals = verify_signals(frames, prices, bank, registered, variant, data)
                item = {**metrics_for(data, benchmark), 'variant': variant, 'scenario': scenario,
                    'ledger': ledger, 'signals': signals, 'issues': data['issue_totals']}
                path = directory / (key+'.json')
                write_json_atomic(path, data)
                pd.DataFrame(data['report']).to_csv(directory/(key+'_report.csv'), index=False)
                item.update(artifact=path.name, sha256=digest(path), seconds=time.monotonic()-began)
                result['cases'][key] = item
                write_json_atomic(directory/'results.json', result)
                print(json.dumps({'case': key, 'cagr': item['metrics']['strategy']['cagr'],
                    'daily_drawdown': item['metrics']['strategy']['mean_daily_drawdown'],
                    'max_drawdown': item['metrics']['strategy']['max_drawdown'],
                    'checks': item['metrics']['checks'], 'seconds': item['seconds']}), flush=True)
        require(len(result['cases']) == 15, '情景未全部完成')
        result['selection_gate'] = decision_summary(result['cases'])
        check_inputs(registered)
        write_json_atomic(directory/'results.json', result)
        write_json_atomic(directory/'completion.json', {'cases': 15,
            'registration_sha256': digest(directory/'registration.json'), 'results_sha256': digest(directory/'results.json'),
            'completed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
        print(json.dumps({'completed': str(directory), **result['selection_gate']}), flush=True)
    except Exception as exc:
        write_json_atomic(directory/'failure.json', {'error': str(exc), 'completed_cases': list(result['cases']),
            'failed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
        raise


def verify(directory):
    directory = Path(directory).resolve()
    registered, result, completion = (read(directory/f) for f in ('registration.json', 'results.json', 'completion.json'))
    require(len(result['cases']) == completion['cases'] == 15 and digest(directory/'registration.json') == completion['registration_sha256']
            and digest(directory/'results.json') == completion['results_sha256'], '归档未完整完成或结果被修改')
    check_inputs(registered)
    frames = PrimaryMarketDays(corrected_market())
    prices, _, bank, feature_proof = feature_bank(frames)
    require(feature_proof == result['feature_verification'], '独立特征重建不同')
    benchmark = pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True)
    outputs, days, signals, maximum = {}, 0, 0, 0.
    for key, item in result['cases'].items():
        path = directory / item['artifact']
        require(digest(path) == item['sha256'], '情景账本归档被修改')
        data = read(path)
        checked = inspect_case(frames, registered, item['scenario'], registered['actions'], data)
        require(checked == item['ledger'], '落盘账本独立重建不符')
        signal_check = verify_signals(frames, prices, bank, registered, item['variant'], data)
        require(signal_check == item['signals'], '落盘信号独立重建不符')
        measured = metrics_for(data, benchmark)
        require(all(measured[k] == item[k] for k in measured), '收益回撤或年度指标不能由净值重建')
        validate_state(data['final_state'])
        require(len(data['state_hashes']) == checked['days'], '每日JSON状态恢复数量不符')
        days += checked['days']; signals += signal_check['signals_verified']
        maximum = max(maximum, checked['maximum_absolute_numeric_error'])
        outputs[path.relative_to(BASE).as_posix()] = digest(path)
        print(json.dumps({'verified': key}), flush=True)
    gate = decision_summary(result['cases'])
    require(gate == result['selection_gate'], '主方案和邻域准入判断不符')
    proof = {'verified_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'cases_verified': 15,
        'daily_rows_reconstructed': days, 'signals_reconstructed': signals, 'maximum_numeric_error': maximum,
        'input_hashes_verified': len(registered['input_hashes']), 'artifact_hashes': outputs,
        'feature_verification': feature_proof, 'selection_gate': gate, **completion}
    write_json_atomic(directory/'verification.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k != 'artifact_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register', action='store_true')
    group.add_argument('--run', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run) if args.run else verify(args.verify)
