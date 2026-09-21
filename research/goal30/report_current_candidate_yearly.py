"""重新回放当前冻结候选，并独立计算近十年逐年收益和年内最大回撤。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.audit_group_outcome import check_metrics
from research.goal30.audit_redeployment_outcome import benchmark_from_raw
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import START, END, corrected_market, read
from research.goal30.run_dividend_payment_replay import OUT as SOURCE_ROOT
from research.goal30.run_group_diversification import check_inputs
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import PrimaryMarketDays, drive, inspect_case


def annual_table(strategy, benchmark):
    require(strategy.index.equals(benchmark.index), '策略与基准日期不同')
    result = []
    for year in sorted(set(strategy.index.year)):
        index = np.flatnonzero(strategy.index.year == year)
        start = max(0, index[0]-1)
        row = {'year': int(year), 'first_session': str(strategy.index[index[0]].date()),
            'last_session': str(strategy.index[index[-1]].date()),
            'baseline_date': str(strategy.index[start].date()), 'sessions': len(index),
            'partial_year': year in (strategy.index[0].year, strategy.index[-1].year)}
        for name, series in [('strategy', strategy), ('sp500_cny_total', benchmark)]:
            block = series.iloc[start:index[-1]+1].to_numpy(float)
            row[name+'_return'] = float(block[-1]/block[0]-1)
            # 一套使用逐日高点扫描；再以pandas累计高点独立核对。
            peak, maximum = block[0], 0.
            for value in block:
                peak = max(peak, value)
                maximum = max(maximum, 1-value/peak)
            row[name+'_max_drawdown'] = float(maximum)
            check = pd.Series(block)
            require(abs(maximum-float((1-check/check.cummax()).max())) < 1e-12, '年度回撤扫描不同')
        row['return_difference_pp'] = (row['strategy_return']-row['sp500_cny_total_return'])*100
        result.append(row)
    for name, series in [('strategy', strategy), ('sp500_cny_total', benchmark)]:
        growth = float(np.prod([1+r[name+'_return'] for r in result]))
        require(abs(growth-series.iloc[-1]/series.iloc[0]) < 1e-12, '年度复合收益不能闭合到全期')
    return result


def run():
    source = BASE/read(SOURCE_ROOT/'latest_registration.json')['directory']
    registration, old_results, proof = (read(source/n) for n in ('registration.json', 'results.json', 'verification.json'))
    selected_path = BASE/'research/ex_nasdaq/selected_strategy.json'
    selected = read(selected_path)
    require(registration['candidate'] == selected['candidate'] and proof['passed'], '当前候选与已核验控制不同')
    item = old_results['cases']['fixed_control__base']
    old_path = source/item['artifact']
    require(digest(old_path) == item['sha256'], '旧控制账本变化')
    inputs = {**registration['input_hashes'], **proof['artifact_hashes']}
    for p in (selected_path, source/'verification.json', Path(__file__)):
        inputs[p.relative_to(BASE).as_posix()] = digest(p)
    check_inputs({'input_hashes': inputs})
    directory = BASE/'output/goal30/current_candidate_yearly'/pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory/'scope.json', {'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'candidate': registration['candidate'], 'policy': registration['fixed_policy'], 'scenario': item['scenario'],
        'source': source.relative_to(BASE).as_posix(), 'start': START, 'end': END,
        'intent': 'Rerun the already selected candidate without tuning or selecting another configuration; report all calendar-year slices.',
        'input_hashes': inputs})
    frames = PrimaryMarketDays(corrected_market())
    dates = frames.days[(frames.days >= START) & (frames.days <= END)]
    data = drive(frames, {'candidate': registration['candidate'], 'policy': registration['fixed_policy']},
                 registration['actions'], item['scenario'], dates)
    ledger = inspect_case(frames, {**registration, 'policy': registration['fixed_policy']},
                          item['scenario'], registration['actions'], data)
    require(ledger == item['ledger'], '重新回放的账本核验不同')
    require(data == read(old_path), '重新执行结果与已核验固定候选不同')
    path = directory/'replayed_base.json'; write_json_atomic(path, data)
    reloaded = read(path)
    require(reloaded == data, '落盘账本不同')
    report = pd.DataFrame(reloaded['report']).set_index('date'); report.index = pd.to_datetime(report.index)
    report.to_csv(directory/'daily_report.csv')
    benchmark, benchmark_files = benchmark_from_raw()
    m1 = check_metrics(item['metrics']['strategy'], report.account)
    m2 = check_metrics(item['metrics']['spy_total_cny'], benchmark)
    rows = annual_table(report.account, benchmark)
    for row in rows:
        year = str(row['year']); old = item['annual_returns'][year]
        require(abs(old['strategy']-row['strategy_return']) < 1e-12 and
                abs(old['spy_total_cny']-row['sp500_cny_total_return']) < 1e-12, '与既有年度收益不符')
        if not row['partial_year']:
            require(abs(item['metrics']['strategy']['annual_max_drawdowns'][year]-row['strategy_max_drawdown']) < 1e-12 and
                    abs(item['metrics']['spy_total_cny']['annual_max_drawdowns'][year]-row['sp500_cny_total_max_drawdown']) < 1e-12,
                    '与既有完整年回撤不符')
    csv = directory/'annual_comparison.csv'; pd.DataFrame(rows).to_csv(csv, index=False, encoding='utf-8-sig')
    values = []
    for row in rows:
        label = str(row['year'])
        if row['year'] == 2016: label += '（9月12日起）'
        if row['year'] == 2026: label += '（截至9月11日）'
        values.append(f"| {label} | {row['strategy_return']:.2%} | {row['strategy_max_drawdown']:.2%} | {row['sp500_cny_total_return']:.2%} | {row['sp500_cny_total_max_drawdown']:.2%} | {row['return_difference_pp']:+.2f} |")
    table = '\n'.join(['| 年份 | 策略收益率 | 策略最大回撤 | 标普500收益率 | 标普500最大回撤 | 收益差（百分点） |',
                       '| --- | ---: | ---: | ---: | ---: | ---: |', *values])
    markdown = f'''# 当前冻结候选与标普500：近十年逐年比较

本次重新回放当前保留的通信ETF（515880）60%、黄金ETF（518880）35%、现金5%候选，通信尚未上市或不具备资格时用沪深300ETF（510300）替代；替代ETF也须符合当时资格。它是当前保留研究候选，尚无同时满足收益和日均回撤要求的已验收策略，不能称为已证明的全局最优或已获投资准入。

区间：2016-09-12—2026-09-11，2428个共同中国交易日，从100万元现金开始。每日收盘检查资格和持仓偏离，按3个百分点漂移带触发调整，下一交易日开盘执行；按原规则处理30%每日共享换手预算、整手和流动性限制。策略净值扣除佣金、滑点和冲击成本，并处理拆分、分红权益及公告支付模型。实际持仓不保证时刻等于目标权重。

基准为人民币计价的SPY含分红总收益，代表标普500投资回报；由原始SPY调整收盘价乘同期美元兑人民币汇率重建，按共同中国交易日对齐。它不是美元价格指数，未计个人跨境投资的税费和交易成本。

{table}

年度收益以此前一年最后一个交易日收盘净值为起点；2016年使用回测起点。年度最大回撤在每年重新计高点，并纳入上一年末净值，按收盘净值计算。回撤列用正数表示损失，越小越好。首尾两行不足完整自然年，因此近十年区间涉及11个年份标签。

| 全区间指标 | 策略 | 标普500人民币总收益 |
| --- | ---: | ---: |
| 复合年化收益率（CAGR） | {m1['cagr']:.2%} | {m2['cagr']:.2%} |
| 累计收益率 | {m1['cumulative_return']:.2%} | {m2['cumulative_return']:.2%} |
| 全期最大回撤 | {m1['max_drawdown']:.2%} | {m2['max_drawdown']:.2%} |
| 全交易日日均回撤 | {m1['mean_daily_drawdown']:.2%} | {m2['mean_daily_drawdown']:.2%} |
| 完整年度最大回撤均值（2017—2025） | {m1['mean_annual_max_drawdown']:.2%} | {m2['mean_annual_max_drawdown']:.2%} |

收益达到基准，但日均回撤较大，且全期最大回撤超过项目20%门槛。2025年贡献显著；这是在已查看历史上选出的候选，仍缺独立样本外及126日真实前瞻证据。年度回撤均值较低不能替代日均回撤达标。

复跑的全部账本、成交、信号和期末状态与最新已验收固定控制完全相同；独立复核2428个日账本行。年度收益连乘与全期累计收益闭合，年度最大回撤使用两种独立计算核对。

[下载年度比较CSV](../{csv.relative_to(BASE).as_posix()})。本次归档：`{directory.relative_to(BASE).as_posix()}`。
'''
    doc = BASE/'docs/SP500_CURRENT_CANDIDATE_YEARLY.md'; doc.write_text(markdown, encoding='utf-8')
    check_inputs({'input_hashes': inputs})
    artifact_files = [path, directory/'daily_report.csv', csv, doc, directory/'scope.json', *benchmark_files]
    output = {'passed': True, 'rerun_exactly_equals_verified_control': True, 'ledger': ledger,
        'start': START, 'end': END, 'observations': len(dates), 'annual_rows': rows,
        'strategy_metrics': m1, 'benchmark_metrics': m2, 'input_hashes': inputs,
        'artifact_hashes': {p.relative_to(BASE).as_posix(): digest(p) for p in artifact_files},
        'investment_admission': False, 'full_goal_achieved': False}
    write_json_atomic(directory/'verification.json', output)
    print(json.dumps({'directory': directory.relative_to(BASE).as_posix(), 'ledger_rows': ledger['days'],
        'trade_count': len(data['trades']), 'rerun_identical': True, 'annual_rows': len(rows)}, ensure_ascii=False), flush=True)
    print(table, flush=True)


if __name__ == '__main__': run()
