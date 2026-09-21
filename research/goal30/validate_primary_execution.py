"""登记、回放并独立验收全部已知事件和三个完整历史控制情景。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE, comparison
from research.goal30.bond_dividend_evidence import digest
from research.goal30.daily_simulation import MarketDays
from research.goal30.primary_cash_ledger import EventCalendar, freeze_next, new_state, settle_day, validate_state
from research.goal30.primary_research_series import ROOT, SOURCE, START, END, action_day, corrected_market, read, rebuild_series
from research.goal30.reconcile_primary_execution import reconcile
from research.goal30.turnover_budget import require

REFERENCE = BASE / 'output/goal30/turnover_replay/20260913-231958-9303749a'
CODE = ('primary_research_series.py', 'primary_cash_ledger.py', 'reconcile_primary_execution.py',
        'test_primary_execution.py', 'validate_primary_execution.py')
SCENARIOS = [
    {'name': 'base', 'slippage_multiplier': 1., 'payment_delay_sessions': 0, 'uncertain_payment_extra_sessions': 1},
    {'name': 'slippage_2', 'slippage_multiplier': 2., 'payment_delay_sessions': 0, 'uncertain_payment_extra_sessions': 1},
    {'name': 'payment_delay_5', 'slippage_multiplier': 1., 'payment_delay_sessions': 5, 'uncertain_payment_extra_sessions': 1}]


class PrimaryMarketDays(MarketDays):
    def __init__(self, market):
        super().__init__(market)
        # 最多5日沿用上次指数作估值/风险统计；不生成可成交开盘价或新增原始行情行。
        self.signal_prices = market.wide('adj_close').ffill(limit=5)
        self.returns = self.signal_prices.pct_change(fill_method=None)


def register():
    manifest = read(ROOT / 'series_manifest.json')
    inputs = {**manifest['input_hashes'], **manifest['artifact_hashes']}
    files = [ROOT / 'series_manifest.json', REFERENCE / 'registration.json',
             REFERENCE / 'benchmark_fixed_control__base.csv']
    files += [Path(__file__).with_name(name) for name in CODE]
    files += [REFERENCE / f'report_fixed_control__{s["name"]}.csv' for s in SCENARIOS]
    inputs.update({p.relative_to(BASE).as_posix(): digest(p) for p in files})
    for path, expected in inputs.items():
        require(digest(BASE / path) == expected, f'登记前输入变化：{path}')
    events = read(ROOT / 'actions.json')['events']
    old = read(REFERENCE / 'registration.json')
    directory = ROOT / 'validation' / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'registered_code').mkdir()
    for name in CODE:
        shutil.copyfile(Path(__file__).with_name(name), directory / 'registered_code' / name)
    registration = {'registered_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'input_hashes': inputs,
        'actions': events, 'candidate': old['execution_adapter_candidate'], 'policy': old['policy'],
        'scenarios': SCENARIOS, 'start': START, 'end': END,
        'cases': {'single_event_windows': 201, 'full_history_fixed_controls': 3},
        'protocol': {
            'scope': 'Every one of 67 in-window events under all 3 scenarios. Two warmup-only dividends enter signals but not execution.',
            'windows': 'Start eight sessions before record; end eight sessions after announced pay/ex date plus scenario delay and uncertain-payment buffer. Synthetic 80% target in event ETF until effective date, then exit. Execution fixtures, not strategy candidates.',
            'full_controls': 'Unchanged selected fixed candidate, all three full-history scenarios, signals recomputed from corrected past series and daily account state.',
            'persistence': 'Serialize and restore finite JSON state after every session; save next-session orders and state hash.',
            'cash_model': 'Historical only. Pay after close on announced date plus model delay; uncertain transfer roles receive one extra session. No claim of actual receipt.',
            'independent_audit': 'Reconstruct holdings with Fraction rounding and cash from fills. Verify entitlement stage, every value/cost/turnover row, final state, all action stages and nonzero event exposure. Payments beyond the observed calendar must remain receivable; never extend prices or invent future payment observations.',
            'signal_causality': 'Recompute full price prefixes immediately before every split; future bars/actions may not change earlier values.',
            'promotion': 'No production pointer or automation change. Model sessions do not count toward 126 real-forward days.'}}
    write_json_atomic(directory / 'registration.json', registration)
    write_json_atomic(ROOT / 'latest_validation.json', {'directory': directory.relative_to(BASE).as_posix(),
                                                       'registration_sha256': digest(directory / 'registration.json')})
    print(json.dumps({'registered': directory.relative_to(BASE).as_posix(), 'cases': 204, 'inputs': len(inputs)}, ensure_ascii=False), flush=True)


def verify_series(actions):
    series = pd.read_parquet(ROOT / 'signal_series.parquet')
    raw = pd.read_parquet(SOURCE / 'market_snapshot.parquet')
    raw = raw[raw.symbol.isin(series.symbol.unique())].sort_values(['symbol', 'date'])
    event_map = {(e['symbol'], action_day(e)): e for e in actions}
    actual = series.set_index(['symbol', 'date'])
    require(len(actual) == len(raw) and actual.index.is_unique, '信号主键不同')
    max_error, used = 0., set()
    for symbol, frame in raw.groupby('symbol'):
        index, previous = 100., None
        for row in frame.itertuples():
            e = event_map.get((symbol, str(row.date.date())))
            ratio, dividend = 1., 0.
            if e:
                used.add(e['event_id'])
                if e['type'] == 'split':
                    ratio = float(e['ratio_decimal'])
                else:
                    dividend = float(e['cash_per_share_decimal'])
            if previous is not None:
                index *= (row.close * ratio + dividend) / previous
            point = actual.loc[(symbol, row.date)]
            require(point['split_ratio'] == ratio and point['cash_dividend_per_share'] == dividend, '信号行动数值不符')
            error = abs(point['signal_total_return_index'] - index)
            max_error = max(max_error, error)
            require(error < 1e-9, '总收益指数不能由原价及行动独立重建')
            previous = row.close
    require(used == {e['event_id'] for e in actions}, '总收益重建漏行动')
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    cutoffs = {dates[dates.searchsorted(e['effective_date']) - 1] for e in actions if e['type'] == 'split'}
    cutoffs.add(pd.Timestamp('2016-01-19'))
    for cutoff in sorted(cutoffs):
        rebuilt, _ = rebuild_series(raw[raw.date <= cutoff], actions)
        expected = series[series.date <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(rebuilt, expected)
    return {'rows': len(series), 'symbols': series.symbol.nunique(), 'events': len(used),
            'prefixes_verified': len(cutoffs), 'maximum_index_error': max_error}


def drive(frames, selected, actions, scenario, dates, target_symbol=None, exit_date=None):
    catalog = EventCalendar(actions, frames.days, payment_mode='announced_schedule_model',
        payment_delay_sessions=scenario['payment_delay_sessions'],
        uncertain_payment_extra_sessions=scenario['uncertain_payment_extra_sessions'])
    policy = selected['policy']
    state = new_state(policy['initial_capital'], frames.context(dates[0])['previous_date'])
    data = {'report': [], 'trades': [], 'actions': [], 'holdings': [], 'signals': [], 'state_hashes': [], 'pending': {}}
    for i, day in enumerate(dates):
        context = frames.context(day)
        data['pending'][context['date']] = state['pending_orders']
        state, output = settle_day(state, context, policy, catalog, slippage_multiplier=scenario['slippage_multiplier'])
        data['report'].append(output['report'])
        for kind in ('trades', 'actions', 'holdings'):
            data[kind].extend(output[kind])
        if i + 1 < len(dates):
            if target_symbol:
                targets = {target_symbol: .8} if str(day.date()) < exit_date else {}
            else:
                targets = frames.fixed_targets(selected, state, START)
            data['signals'].append({'date': str(day.date()), 'target_weights': targets,
                                    'fixture': bool(target_symbol)})
            state = freeze_next(state, targets, context['close'], frames.adv.loc[day].to_dict(), policy, dates[i+1])
        state = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
        validate_state(state)
        data['state_hashes'].append({'date': context['date'], 'state_sha256': state['state_sha256']})
    data['final_state'] = state
    return data


def inspect_case(frames, registered, scenario, events, data):
    return reconcile(frames, events, registered['policy'], scenario,
        *(pd.DataFrame(data[k]) for k in ('report', 'trades', 'actions', 'holdings')),
        data['final_state'], data['pending'])


def verify_event_exposure(event, checked, data, scenario, calendar):
    key = event['event_id']
    if event['type'] == 'split':
        require(key in checked['nonzero_splits'], '窗口没有实际持仓经历拆分，不能计为覆盖')
        return 'nonzero_split'
    require(key in checked['nonzero_dividend_rights'], '窗口没有非零分红权益')
    dates = [str(d.date()) for d in calendar if str(d.date()) >= event['announced_payment_date']]
    uncertain = event['payment_semantics'] in {
        'custodian_outgoing_transfer_actual_broker_receipt_unconfirmed',
        'clearing_participant_transfer_actual_broker_receipt_unconfirmed',
        'announced_distribution_date_actual_broker_receipt_unconfirmed',
        'secondary_schedule_venue_not_primary_verified'}
    delay = scenario['payment_delay_sessions'] + (scenario['uncertain_payment_extra_sessions'] if uncertain else 0)
    if delay < len(dates) and dates[delay] <= data['final_state']['last_date']:
        require(checked['payments'].get(key, 0) > 0, '窗口内应有模型支付而未发生')
        return 'nonzero_model_payment'
    require(key not in checked['payments'] and data['final_state']['receivable'].get(key, 0) > 0,
            '支付日在历史截止日之外，必须保留非零应收而不得提前入现金')
    return 'nonzero_receivable_payment_beyond_observation'


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'validation'), '不在登记目录')
    registered = read(directory / 'registration.json')
    for path, expected in registered['input_hashes'].items():
        require(digest(BASE / path) == expected, f'登记后输入变化：{path}')
    with (directory / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}, stream)
    events = registered['actions']
    frames = PrimaryMarketDays(corrected_market())
    selected = {'candidate': registered['candidate'], 'policy': registered['policy']}
    results = {'cases': {}, 'series': verify_series(events), 'investment_admission': False, 'full_goal_achieved': False}
    try:
        for ordinal, event in enumerate([e for e in events if not e['signal_only']], 1):
            for scenario in registered['scenarios']:
                start_i = max(1, frames.days.searchsorted(pd.Timestamp(event['record_date'])) - 8)
                endpoint = event.get('announced_payment_date', event.get('effective_date'))
                end_i = min(len(frames.days)-1, frames.days.searchsorted(pd.Timestamp(endpoint)) + 9 + scenario['payment_delay_sessions'])
                dates = frames.days[start_i:end_i+1]
                key = f'event_{ordinal:02d}__{scenario["name"]}'
                data = drive(frames, selected, [event], scenario, dates, event['symbol'], action_day(event))
                checked = inspect_case(frames, registered, scenario, [event], data)
                exposure = verify_event_exposure(event, checked, data, scenario, frames.days)
                path = directory / (key + '.json')
                write_json_atomic(path, data)
                results['cases'][key] = {'kind': 'execution_fixture_not_strategy', 'event_id': event['event_id'],
                    'scenario': scenario, 'verification': checked, 'exposure_check': exposure,
                    'artifact': path.name, 'sha256': digest(path)}
            if ordinal % 10 == 0 or ordinal == 67:
                write_json_atomic(directory / 'progress.json', {'event_windows_completed': ordinal * 3, 'full_controls_completed': 0})
                print(json.dumps({'event_windows_completed': ordinal * 3, 'total': 201}), flush=True)
        benchmark = pd.read_csv(REFERENCE / 'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True)
        dates = frames.days[(frames.days >= START) & (frames.days <= END)]
        for scenario in registered['scenarios']:
            key = 'fixed_control__' + scenario['name']
            began = time.monotonic()
            data = drive(frames, selected, events, scenario, dates)
            checked = inspect_case(frames, registered, scenario, events, data)
            report = pd.DataFrame(data['report']).set_index('date')
            report.index = pd.to_datetime(report.index)
            metrics = comparison(report.account, benchmark.spy_total_cny)
            metrics['checks']['maximum_drawdown_within_20pct'] = metrics['strategy']['max_drawdown'] <= .20
            metrics['checks']['joint_admission_performance'] = metrics['checks']['daily_definition_target_met'] and metrics['checks']['maximum_drawdown_within_20pct']
            old_report = pd.read_csv(REFERENCE / f'report_{key}.csv', index_col=0, parse_dates=True)
            require(old_report.index.equals(report.index), '固定控制日期与旧归档不同')
            prior_difference = float((old_report.account - report.account).abs().max())
            path = directory / (key + '.json')
            write_json_atomic(path, data)
            report.to_csv(directory / (key + '_report.csv'))
            results['cases'][key] = {'kind': 'fixed_candidate_control_not_new_strategy', 'scenario': scenario,
                'verification': checked, 'metrics': metrics, 'maximum_account_difference_from_old': prior_difference,
                'artifact': path.name, 'sha256': digest(path), 'seconds': time.monotonic() - began}
            write_json_atomic(directory / 'progress.json', {'event_windows_completed': 201,
                'full_controls_completed': sum(v['kind'] == 'fixed_candidate_control_not_new_strategy' for v in results['cases'].values())})
            print(json.dumps({'case': key, 'strategy': metrics['strategy'], 'checks': metrics['checks'],
                              'maximum_old_account_difference': prior_difference, 'seconds': time.monotonic() - began}), flush=True)
        require(len(results['cases']) == 204, '未完成全部预登记情景')
        for path, expected in registered['input_hashes'].items():
            require(digest(BASE / path) == expected, f'运行期间输入变化：{path}')
        write_json_atomic(directory / 'results.json', results)
        write_json_atomic(directory / 'completion.json', {'cases': 204, 'results_sha256': digest(directory / 'results.json'),
            'registration_sha256': digest(directory / 'registration.json'), 'qualified_real_forward_sessions': 0})
        print(json.dumps({'completed': directory.relative_to(BASE).as_posix(), 'cases': 204}), flush=True)
    except Exception as exc:
        write_json_atomic(directory / 'failure.json', {'error': str(exc), 'completed_cases': list(results['cases']),
            'failed_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
        raise


def verify(directory):
    directory = Path(directory).resolve()
    registered = read(directory / 'registration.json')
    result = read(directory / 'results.json')
    completion = read(directory / 'completion.json')
    require(digest(directory / 'registration.json') == completion['registration_sha256'] and
            digest(directory / 'results.json') == completion['results_sha256'] and len(result['cases']) == completion['cases'] == 204,
            '验收运行未完整完成或登记/结果被修改')
    for path, expected in registered['input_hashes'].items():
        require(digest(BASE / path) == expected, f'复核输入变化：{path}')
    frames = PrimaryMarketDays(corrected_market())
    events = {e['event_id']: e for e in registered['actions']}
    require(verify_series(registered['actions']) == result['series'], '总收益信号复核不同')
    days, rights, splits, paid, pending = 0, set(), set(), set(), set()
    output_hashes = {}
    for key, case in result['cases'].items():
        path = directory / case['artifact']
        require(digest(path) == case['sha256'], '逐日回放归档改变')
        data = read(path)
        scope = [events[case['event_id']]] if 'event_id' in case else list(events.values())
        checked = inspect_case(frames, registered, case['scenario'], scope, data)
        require(checked == case['verification'], '从落盘数据独立重建与原验收不同')
        if 'event_id' in case:
            exposure = verify_event_exposure(scope[0], checked, data, case['scenario'], frames.days)
            require(exposure == case['exposure_check'], '实际持仓事件覆盖分类不同')
            if exposure == 'nonzero_receivable_payment_beyond_observation':
                pending.add(case['event_id'])
        paid.update(k for k, v in checked['payments'].items() if v > 0)
        days += checked['days']
        rights.update(checked['nonzero_dividend_rights'])
        splits.update(checked['nonzero_splits'])
        output_hashes[path.relative_to(BASE).as_posix()] = digest(path)
        validate_state(data['final_state'])
        require(len(data['state_hashes']) == checked['days'], '逐日恢复数量不符')
    require(len(rights) == 53 and len(splits) == 14, '非零现金权益或拆分覆盖未达到全部67项')
    require(len(paid) == 52 and pending == {'SH512400:cash_dividend:2026-09-09'}, '实际可观察支付与期末应收覆盖不符')
    proof = {'verified_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'cases_verified': 204,
        'daily_rows_reconstructed': days, 'nonzero_cash_events': len(rights), 'nonzero_split_events': len(splits),
        'nonzero_model_payment_events': len(paid), 'payment_beyond_observation': sorted(pending),
        'series': result['series'], 'inputs_verified': len(registered['input_hashes']), 'artifact_hashes': output_hashes,
        'registration_sha256': digest(directory / 'registration.json'), 'results_sha256': digest(directory / 'results.json'),
        'investment_admission': False, 'full_goal_achieved': False, 'qualified_real_forward_sessions': 0}
    write_json_atomic(directory / 'verification.json', proof)
    print(json.dumps({k: v for k, v in proof.items() if k != 'artifact_hashes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register', action='store_true')
    group.add_argument('--run', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.register:
        register()
    elif args.run:
        run(args.run)
    else:
        verify(args.verify)
