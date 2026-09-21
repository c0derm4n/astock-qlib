"""核对平均回撤口径和严格近十年区间；保留旧候选的全部筛查结果。"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.replay import load_selected

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'output/goal30'
SELECTED = BASE / 'research/ex_nasdaq/selected_strategy.json'


def wealth_metrics(wealth):
    wealth = wealth.astype(float)
    if (not isinstance(wealth.index, pd.DatetimeIndex) or len(wealth) < 2
            or not wealth.index.is_monotonic_increasing or not wealth.index.is_unique
            or not np.isfinite(wealth).all() or (wealth <= 0).any()):
        raise ValueError('净值必须为日期递增、无重复、完整且为正的序列')
    days = (wealth.index[-1] - wealth.index[0]).days
    if days <= 0:
        raise ValueError('净值区间不足一天')
    normalized = wealth / wealth.iloc[0]
    drawdown = 1 - normalized / normalized.cummax()
    annual = {}
    # 首尾年可能不完整，统一排除；每年纳入上一年最后收盘作为年初基准。
    for year in range(wealth.index[0].year + 1, wealth.index[-1].year):
        positions = np.flatnonzero(wealth.index.year == year)
        if not len(positions):
            raise ValueError(f'缺少{year}年净值')
        block = wealth.iloc[positions[0] - 1:positions[-1] + 1]
        annual[str(year)] = float((1 - block / block.cummax()).max())
    return {'start': str(wealth.index[0].date()), 'end': str(wealth.index[-1].date()),
            'observations': len(wealth), 'cumulative_return': float(normalized.iloc[-1] - 1),
            'cagr': float(normalized.iloc[-1] ** (365.25 / days) - 1),
            'mean_daily_drawdown': float(drawdown.mean()),
            'mean_annual_max_drawdown': float(np.mean(list(annual.values()))) if annual else None,
            'max_drawdown': float(drawdown.max()), 'annual_max_drawdowns': annual}


def comparison(strategy, benchmark):
    if not strategy.index.equals(benchmark.index):
        raise ValueError('策略和基准必须使用相同交易日；不能静默删除缺失日期')
    a, b = wealth_metrics(strategy), wealth_metrics(benchmark)
    annual_ok = (a['mean_annual_max_drawdown'] is not None
                 and a['mean_annual_max_drawdown'] <= b['mean_annual_max_drawdown'] + 1e-12)
    daily_ok = a['mean_daily_drawdown'] <= b['mean_daily_drawdown'] + 1e-12
    cagr_ok = a['cagr'] >= b['cagr'] - 1e-12
    return {'strategy': a, 'spy_total_cny': b,
            'checks': {'cagr_at_least_benchmark': cagr_ok, 'mean_annual_drawdown_no_worse': annual_ok,
                       'mean_daily_drawdown_no_worse': daily_ok,
                       'annual_definition_target_met': cagr_ok and annual_ok,
                       'daily_definition_target_met': cagr_ok and daily_ok,
                       'both_definitions_target_met': cagr_ok and annual_ok and daily_ok}}


def read_pair(report, benchmark):
    a = pd.read_csv(report, index_col=0, parse_dates=True)['account']
    b = pd.read_csv(benchmark, index_col=0, parse_dates=True)['spy_total_cny']
    if not a.index.equals(b.index):
        raise ValueError(f'基准日期不匹配：{report}')
    return a, b


def audit(*, restart=False):
    selected, validation = load_selected(SELECTED)
    a, b = read_pair(validation / 'report_base_long.csv', validation / 'benchmark_base_long.csv')
    cutoff = a.index[-1] - pd.DateOffset(years=10)
    ten_start = a.index[a.index >= cutoff][0]
    directory = OUT / 'audits' / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    input_paths = [SELECTED, Path(__file__), validation / 'report_base_long.csv',
                   validation / 'benchmark_base_long.csv']
    result = {'schema_version': 1, 'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
              'investment_admission': False, 'historical_inputs_verified': True,
              'average_drawdown_definition': 'both_reported_user_preference_pending',
              'ten_year_start': str(ten_start.date()), 'ten_year_end': str(a.index[-1].date()),
              'full_history': comparison(a, b),
              'ten_year_continuation': comparison(a.loc[ten_start:], b.loc[ten_start:]),
              'continuation_note': '持有状态延续切片；不代表十年前从现金重新建仓',
              'selection_bias_note': '95个已查看历史配置的诊断；不能作为独立样本外确认',
              'existing_candidates': []}
    for run in sorted((BASE / 'output/sp500_ex_nasdaq/runs').iterdir()):
        if not run.is_dir():
            continue
        registry_path = run / 'registry.json'
        if not (run / 'completion.json').exists():
            raise ValueError(f'候选运行未完成：{run}')
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        input_paths.append(registry_path)
        for candidate in registry['candidates']:
            report = run / f"report_{candidate['name']}_long.csv"
            benchmark = run / f"benchmark_{candidate['name']}_long.csv"
            if not benchmark.exists():
                benchmark = run / 'benchmark_long.csv'
            x, y = read_pair(report, benchmark)
            if x.index[0] > ten_start or x.index[-1] != a.index[-1]:
                raise ValueError(f'候选未覆盖统一十年区间：{report}')
            metrics = comparison(x.loc[ten_start:], y.loc[ten_start:])
            result['existing_candidates'].append({'run': run.name, 'candidate': candidate['name'], **metrics})
            input_paths.extend([report, benchmark])
    result['screening_counts'] = {'registered_configurations': len(result['existing_candidates']), **{
        key: sum(row['checks'][key] for row in result['existing_candidates'])
        for key in ('annual_definition_target_met', 'daily_definition_target_met', 'both_definitions_target_met')}}
    if restart:
        from research.ex_nasdaq.core import market30, execute
        from research.sp500.validate import relative_metrics, ledger_audit
        market = market30(BASE / selected['source'])
        report, trades, issues = execute(market, selected['candidate'], selected['policy'], ten_start)
        _, benchmark = relative_metrics(report, BASE / selected['benchmark'], selected['comparison'])
        result['ten_year_cash_restart'] = comparison(report.account, benchmark.spy_total_cny)
        result['restart_ledger'] = ledger_audit(market, report, trades, selected['policy'],
                                               selected['policy']['initial_capital'])
        result['restart_issues'] = issues
        report.to_csv(directory / 'report_cash_restart.csv')
        trades.to_csv(directory / 'trades_cash_restart.csv', index=False)
        benchmark.to_csv(directory / 'benchmark_cash_restart.csv')
    result['input_hashes'] = {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in input_paths}
    write_json_atomic(directory / 'results.json', result)
    lines = ['# 30只ETF目标口径核验', '',
             '回撤为正的损失幅度，越小越好。基准为含分红、折算人民币的SPY；双方使用相同中国交易日。', '',
             '| 口径 | 对象 | 年化收益 | 日均回撤 | 年度最大回撤均值 | 全区间最大回撤 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    labels = {'full_history': '原完整区间', 'ten_year_continuation': '十年持仓延续',
              'ten_year_cash_restart': '十年现金重建'}
    for key, label in labels.items():
        if key not in result:
            continue
        for name, display in [('strategy', '策略'), ('spy_total_cny', '人民币SPY')]:
            row = result[key][name]
            lines.append(f"| {label} | {display} | {row['cagr']:.2%} | {row['mean_daily_drawdown']:.2%} | "
                         f"{row['mean_annual_max_drawdown']:.2%} | {row['max_drawdown']:.2%} |")
    lines += ['', f"严格十年共同区间：{ten_start.date()}—{a.index[-1].date()}。年度均值包含2017—2025年；首尾不完整年度排除。",
              '', '已登记95个配置全部保留，筛查只作历史诊断，不消除行业选择、参数选择或2025年依赖。',
              '', '筛查计数：' + json.dumps(result['screening_counts'], ensure_ascii=False), '',
              '**整体目标尚未完成。** 平均回撤口径待确认；历史数据审计与126日真实前瞻仍未通过，不能输出实盘买卖指令。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json_atomic(OUT / 'latest_audit.json', {'directory': str(directory),
                      'results_sha256': hashlib.sha256((directory / 'results.json').read_bytes()).hexdigest()})
    print(json.dumps({'directory': str(directory), 'screening_counts': result['screening_counts'],
                      'restart': result.get('ten_year_cash_restart', {}).get('checks')}, ensure_ascii=False, indent=2))
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restart', action='store_true', help='另从十年起点现金重建并审计账本')
    audit(restart=parser.parse_args().restart)
