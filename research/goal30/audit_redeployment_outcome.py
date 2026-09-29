"""从原始基准及配对账本核对资金重新分配研究的表现和成本。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.audit_group_outcome import check_metrics
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import START, END, read
from research.goal30.run_redeployment import OUT, PRIOR
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import REFERENCE
from research.goal30.verify_turnover_replay import independent_metrics


def benchmark_from_raw():
    source = BASE/read(REFERENCE/'registration.json')['benchmark']
    spy = pd.read_csv(source/'spy.csv', index_col='date', parse_dates=True).adjusted_close
    fx = pd.read_csv(source/'usdcny.csv', index_col='date', parse_dates=True).close
    market = BASE/'output/runs/20260912-220345-f20ea3f5/market_snapshot.parquet'
    dates = pd.DatetimeIndex(sorted(pd.read_parquet(market, columns=['date']).date.unique()))
    dates = dates[(dates >= START) & (dates <= END)]
    canonical = spy.reindex(dates, method='ffill', tolerance=pd.Timedelta(days=7))*fx.reindex(
        dates, method='ffill', tolerance=pd.Timedelta(days=7))
    require(len(dates) == 2428 and np.isfinite(canonical).all() and (canonical > 0).all(), '共同基准不完整')
    saved = pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True).spy_total_cny
    require(saved.index.equals(dates) and np.max(abs(saved/saved.iloc[0]-canonical/canonical.iloc[0])) < 1e-12,
            '原始SPY和汇率重建的基准不同')
    return canonical, [market, source/'spy.csv', source/'usdcny.csv', REFERENCE/'registration.json']


def checks_for(a, b, full=True):
    cagr = a['cagr'] >= b['cagr']-1e-12
    daily = a['mean_daily_drawdown'] <= b['mean_daily_drawdown']+1e-12
    annual = a['mean_annual_max_drawdown'] <= b['mean_annual_max_drawdown']+1e-12
    result = {'cagr_at_least_benchmark': cagr, 'mean_annual_drawdown_no_worse': annual,
        'mean_daily_drawdown_no_worse': daily, 'annual_definition_target_met': cagr and annual,
        'daily_definition_target_met': cagr and daily, 'both_definitions_target_met': cagr and daily and annual}
    if full:
        result.update(maximum_drawdown_within_20pct=a['max_drawdown'] <= .2,
                      joint_admission_performance=cagr and daily and a['max_drawdown'] <= .2)
    return result


def inspect_performance(directory, item, benchmark, initial):
    path = directory/item['artifact']
    require(digest(path) == item['sha256'], '配对账本指纹不同')
    data = read(path)
    report = pd.DataFrame(data['report']).set_index('date')
    report.index = pd.to_datetime(report.index)
    csv = directory/(path.stem+'_report.csv')
    pd.testing.assert_frame_equal(pd.read_csv(csv, index_col='date', parse_dates=True), report,
                                  check_dtype=False, atol=1e-8, rtol=1e-12)
    require(report.index.equals(benchmark.index) and report.account.iloc[0] == initial, '十年日期或初始资金不同')
    a = check_metrics(item['metrics']['strategy'], report.account)
    b = check_metrics(item['metrics']['spy_total_cny'], benchmark)
    require(item['metrics']['checks'] == checks_for(a, b), '收益回撤标签不同')
    for label, series in [('strategy', report.account), ('spy_total_cny', benchmark)]:
        m = item['metrics'][label]
        require(m['start'] == START and m['end'] == END and m['observations'] == len(benchmark), '指标区间标签不同')
        check_metrics(item['pre_2025_diagnostic'][label], series.loc[:'2024-12-31'])
    early = item['pre_2025_diagnostic']
    require(early['checks'] == checks_for(early['strategy'], early['spy_total_cny'], full=False), '早期诊断标签不同')
    previous_a, previous_b = initial, benchmark.iloc[0]
    require(set(item['annual_returns']) == set(map(str, sorted(set(benchmark.index.year)))), '逐年收益缺年')
    for year in sorted(set(benchmark.index.year)):
        end_a = report.account[report.index.year == year].iloc[-1]
        end_b = benchmark[benchmark.index.year == year].iloc[-1]
        row = item['annual_returns'][str(year)]
        require(abs(row['strategy']-(end_a/previous_a-1)) < 1e-12 and
                abs(row['spy_total_cny']-(end_b/previous_b-1)) < 1e-12 and
                row['partial_year'] == (year in (2016, 2026)), '逐年收益或不足整年标记不同')
        previous_a, previous_b = end_a, end_b
    trades = pd.DataFrame(data['trades'])
    require(len(trades) > 0 and set(trades.side).issubset({'buy', 'sell'}), '成交方向无效')
    trades['date'] = pd.to_datetime(trades.date)
    require(np.isfinite(trades[['commission', 'slippage_cost', 'quantity', 'raw_price']]).all().all() and
            (trades[['commission', 'slippage_cost']] >= 0).all().all(), '费用含无效数值')
    prior_nav = report.account.shift(1, fill_value=initial)
    costs = trades.assign(total_cost=trades.commission+trades.slippage_cost).groupby('date').total_cost.sum().reindex(report.index, fill_value=0)
    notional = trades.assign(raw_notional=trades.quantity*trades.raw_price).groupby('date').raw_notional.sum().reindex(report.index, fill_value=0)
    require(np.max(abs(costs/prior_nav-report.cost)) < 1e-12, '逐日费用与成交不同')
    require(np.max(abs(notional/prior_nav-report.turnover)) < 1e-12, '逐日双边换手与成交不同')
    require(np.max(abs(report.account/prior_nav-1-(report['return']-report.cost))) < 1e-12, '净收益与成本重复或遗漏')
    invested = float((report.value/report.account).mean())
    turnover = float((notional/prior_nav).sum())
    require(abs(invested-item['mean_invested_fraction']) < 1e-12 and
            abs(turnover-item['sum_daily_two_way_turnover']) < 1e-10 and len(trades) == item['trade_count'], '仓位、换手或成交数不同')
    years = (report.index[-1]-report.index[0]).days/365.25
    return {'metrics': a, 'pre_2025_metrics': early['strategy'], 'checks': checks_for(a, b),
        'mean_invested_fraction': invested, 'sum_daily_two_way_turnover': turnover,
        'average_annual_two_way_turnover': turnover/years, 'trade_count': len(trades),
        'total_commission_cny': float(trades.commission.sum()), 'total_slippage_and_impact_cny': float(trades.slippage_cost.sum()),
        'sum_daily_cost_fraction': float((costs/prior_nav).sum()),
        'cash_only_days': int(report.value.eq(0).sum()), 'first_trade': str(trades.date.min().date())}, [path, csv]


def audit(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT), '需使用已登记的本研究目录')
    reg, result, proof = (read(directory/n) for n in ('registration.json', 'results.json', 'verification.json'))
    require(proof['passed'] and proof['cases_verified'] == 15 and not (directory/'failure.json').exists(), '执行核验尚未完整通过')
    hashes = {**reg['input_hashes'], **proof['artifact_hashes']}
    for rel, expected in hashes.items():
        require(digest(BASE/rel) == expected, '冻结输入或产物变化：'+rel)
    for copy in (directory/'registered_code').iterdir():
        require(digest(copy) == digest(Path(__file__).with_name(copy.name)), '登记源码副本不同')
        hashes[copy.relative_to(BASE).as_posix()] = digest(copy)
    expected = {v['name']+'__'+s['name']: (v, s) for v in reg['variants'] for s in reg['scenarios']}
    require(len(expected) == 15 and set(expected) == set(result['cases']), '15个预登记情景未全部覆盖')
    prior = read(PRIOR/'results.json')['cases']
    benchmark, files = benchmark_from_raw()
    rows = {}
    for key, (variant, scenario) in expected.items():
        item = result['cases'][key]
        require(item['variant'] == prior[key]['variant'] == variant and item['scenario'] == prior[key]['scenario'] == scenario,
                '新旧情景参数不同')
        new, f1 = inspect_performance(directory, item, benchmark, reg['policy']['initial_capital'])
        old, f2 = inspect_performance(PRIOR, prior[key], benchmark, reg['policy']['initial_capital'])
        paired = {k+'_delta': new['metrics'][k]-old['metrics'][k] for k in ('cagr', 'mean_daily_drawdown', 'max_drawdown')}
        paired.update({k+'_delta': new[k]-old[k] for k in ('mean_invested_fraction', 'trade_count')})
        paired['two_way_turnover_delta'] = new['sum_daily_two_way_turnover']-old['sum_daily_two_way_turnover']
        require(set(paired) == set(item['paired_comparison']) and all(abs(v-item['paired_comparison'][k]) < 1e-10 for k,v in paired.items()), '配对差额不能由新旧账本重建')
        costs = {k+'_delta': new[k]-old[k] for k in ('total_commission_cny', 'total_slippage_and_impact_cny', 'sum_daily_cost_fraction')}
        rows[key] = {'new': new, 'prior': old, 'paired': paired, 'paired_cost': costs, 'signals': item['signals']}
        files.extend(f1+f2)
    joint = {k: v['new']['checks']['joint_admission_performance'] for k,v in rows.items()}
    primary = all(joint['group4_risk_parity__'+s['name']] for s in reg['scenarios'])
    neighbors = sum(joint[f'group{k}_risk_parity__base'] for k in (3,4,5))
    gate = {'primary_all_scenarios_jointly_pass': primary, 'neighborhood_jointly_pass': neighbors,
        'neighborhood_count': 3, 'eligible_for_further_admission_review': primary and neighbors >= 2,
        'investment_admission': False, 'full_goal_achieved': False, 'qualified_real_forward_sessions': 0}
    require(gate == result['selection_gate'], '主方案与邻域门槛计算不同')
    files += [directory/'verification.json', Path(__file__)]
    hashes.update({p.relative_to(BASE).as_posix(): digest(p) for p in files})
    out = {'passed': True, 'audited_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'same_sessions': len(benchmark), 'cases_verified': len(rows), 'paired_report_rows': len(rows)*len(benchmark)*2,
        'benchmark_from_raw_spy_and_fx': independent_metrics(benchmark), 'cases': rows,
        'full_history_joint_performance_passes': sum(joint.values()), 'primary_and_neighbors_gate': gate,
        'input_and_artifact_hashes': hashes, 'investment_admission': False, 'full_goal_achieved': False}
    write_json_atomic(directory/'outcome_audit.json', out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('cases', 'input_and_artifact_hashes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
