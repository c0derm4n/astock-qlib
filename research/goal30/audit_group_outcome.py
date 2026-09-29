"""独立核对分组研究的原始基准、表现、全部情景及CSV/JSON一致性。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import START, END, read
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import REFERENCE
from research.goal30.verify_turnover_replay import independent_metrics


def check_metrics(actual, series):
    independent = independent_metrics(series)
    for key, value in independent.items():
        if isinstance(value, dict):
            require(set(value) == set(actual[key]), '完整自然年集合不符')
            require(all(abs(value[y]-actual[key][y]) < 1e-12 for y in value), '年度最大回撤不符')
        else:
            require(abs(value-actual[key]) < 1e-12, f'独立指标不同：{key}')
    return independent


def audit(directory):
    directory = Path(directory).resolve()
    registration, results, proof = (read(directory/f) for f in ('registration.json', 'results.json', 'verification.json'))
    require(proof['registration_sha256'] == digest(directory/'registration.json') and
            proof['results_sha256'] == digest(directory/'results.json'), '落盘复核与结果不匹配')
    expected_cases = {v['name']+'__'+s['name']: (v,s) for v in registration['protocol']['variants']
                      for s in registration['protocol']['scenarios']}
    require(set(expected_cases) == set(results['cases']) and len(expected_cases) == 15, '未覆盖全部独立方案/压力组合')
    for file, expected in registration['input_hashes'].items():
        require(digest(BASE/file) == expected, f'原输入变化：{file}')
    for copy in (directory/'registered_code').iterdir():
        require(digest(copy) == digest(Path(__file__).with_name(copy.name)), '执行源码与登记副本不同')
    reference = read(REFERENCE/'registration.json')
    benchmark_dir = BASE/reference['benchmark']
    spy = pd.read_csv(benchmark_dir/'spy.csv', index_col='date', parse_dates=True).adjusted_close
    fx = pd.read_csv(benchmark_dir/'usdcny.csv', index_col='date', parse_dates=True).close
    raw = pd.read_parquet(BASE/'output/runs/20260912-220345-f20ea3f5/market_snapshot.parquet', columns=['date'])
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    dates = dates[(dates >= START) & (dates <= END)]
    canonical = spy.reindex(dates, method='ffill', tolerance=pd.Timedelta(days=7)) * fx.reindex(dates, method='ffill', tolerance=pd.Timedelta(days=7))
    require(np.isfinite(canonical).all() and (canonical>0).all() and len(dates) == 2428, '原始SPY/汇率基准不完整')
    saved = pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True).spy_total_cny
    require(saved.index.equals(dates) and np.max(np.abs(saved/saved.iloc[0]-canonical/canonical.iloc[0])) < 1e-12,
            '沿用的基准不等于原SPY乘同期汇率')
    rows, artifacts = {}, {}
    for key, (variant, scenario) in expected_cases.items():
        item = results['cases'][key]
        require(item['variant'] == variant and item['scenario'] == scenario, '情景登记参数与结果不同')
        data_path = directory/item['artifact']
        require(digest(data_path) == item['sha256'] == proof['artifact_hashes'][data_path.relative_to(BASE).as_posix()], '账本哈希不符')
        data = read(data_path)
        report = pd.DataFrame(data['report']).set_index('date')
        report.index = pd.to_datetime(report.index)
        csv = directory/(key+'_report.csv')
        saved_report = pd.read_csv(csv, index_col='date', parse_dates=True)
        pd.testing.assert_frame_equal(saved_report, report, check_dtype=False, rtol=1e-12, atol=1e-8)
        require(report.index.equals(dates) and report.account.iloc[0] == registration['policy']['initial_capital'], '未从共同十年起点现金开始')
        a = check_metrics(item['metrics']['strategy'], report.account)
        b = check_metrics(item['metrics']['spy_total_cny'], canonical)
        check_metrics(item['pre_2025_diagnostic']['strategy'], report.account.loc[:'2024-12-31'])
        check_metrics(item['pre_2025_diagnostic']['spy_total_cny'], canonical.loc[:'2024-12-31'])
        target = a['cagr'] >= b['cagr']-1e-12 and a['mean_daily_drawdown'] <= b['mean_daily_drawdown']+1e-12
        require(item['metrics']['checks']['daily_definition_target_met'] == target and
                item['metrics']['checks']['joint_admission_performance'] == (target and a['max_drawdown'] <= .2), '联合目标标签不符')
        previous_a, previous_b = report.account.iloc[0], canonical.iloc[0]
        for year in sorted(set(dates.year)):
            endpoint_a = report.account[report.index.year == year].iloc[-1]
            endpoint_b = canonical[canonical.index.year == year].iloc[-1]
            actual = item['annual_returns'][str(year)]
            require(abs(actual['strategy']-(endpoint_a/previous_a-1)) < 1e-12 and
                    abs(actual['spy_total_cny']-(endpoint_b/previous_b-1)) < 1e-12, '逐年收益不同')
            previous_a, previous_b = endpoint_a, endpoint_b
        exposure_by_symbol = {}
        for h in data['holdings']:
            exposure_by_symbol[h['symbol']] = exposure_by_symbol.get(h['symbol'], 0.) + h['value']/report.at[pd.Timestamp(h['date']), 'account']/len(dates)
        rows[key] = {'cagr': a['cagr'], 'daily_drawdown': a['mean_daily_drawdown'], 'max_drawdown': a['max_drawdown'],
            'target_met': target, 'joint_admission_performance': target and a['max_drawdown'] <= .2,
            'mean_invested_fraction': float((report.value/report.account).mean()),
            'average_annual_two_way_turnover': float(report.turnover.sum()/((dates[-1]-dates[0]).days/365.25)),
            'cash_only_days': int(report.value.eq(0).sum()), 'first_trade': min(t['date'] for t in data['trades']) if data['trades'] else None,
            'mean_exposure_by_symbol': exposure_by_symbol, 'trade_count': len(data['trades'])}
        artifacts[csv.relative_to(BASE).as_posix()] = digest(csv)
    benchmark_metrics = independent_metrics(canonical)
    audited = {'audited_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(), 'cases': rows,
        'benchmark_from_raw_spy_and_fx': benchmark_metrics, 'same_sessions': len(dates),
        'report_csv_hashes': artifacts, 'registration_sha256': digest(directory/'registration.json'),
        'results_sha256': digest(directory/'results.json'), 'verification_sha256': digest(directory/'verification.json'),
        'auditor_sha256': digest(Path(__file__)), 'primary_and_neighbors_gate': results['selection_gate'],
        'full_goal_achieved': False, 'investment_admission': False}
    write_json_atomic(directory/'outcome_audit.json', audited)
    print('Independent outcome audit: 15 cases, 2428 common sessions, original SPY/FX, yearly metrics and all report CSVs verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
