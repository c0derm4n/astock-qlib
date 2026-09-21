"""按修订后的逐年回撤门槛汇总动态重选实验，不切换实盘候选。"""
import json
from pathlib import Path
import pandas as pd
from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.explore_dynamic_frequency import ROOT, assess
from research.goal30.audit_redeployment_outcome import benchmark_from_raw
from research.goal30.primary_research_series import read
from research.goal30.run_group_diversification import check_inputs
from research.goal30.turnover_budget import require


def run():
    directory = BASE/read(ROOT/'latest.json')['directory']
    reg,results,done = (read(directory/n) for n in ('registration.json','results.json','completion.json'))
    require(done['cases']==len(results)==9 and done['registration_sha256']==digest(directory/'registration.json')
        and done['results_sha256']==digest(directory/'results.json'),'回放未完整完成或变化')
    check_inputs(reg); benchmark,_ = benchmark_from_raw()
    rows=[]; annual=[]; hashes={}; labels={'monthly':'每月','weekly':'每周（主方案）','daily':'每日'}
    scenarios={'base':'基础','slippage_2':'滑点2倍','payment_delay_5':'支付延迟5日'}
    for key,item in results.items():
        path=directory/item['artifact'];require(digest(path)==item['sha256'],'账本变化')
        check=assess(read(path),benchmark)
        require(all(check[k]==item[k] for k in check),'指标落盘复核不一致')
        m=item['strategy'];f=item['variant']['frequency'];s=item['scenario']['name']
        rows.append(f"| {labels[f]} | {scenarios[s]} | {m['cagr']:.2%} | {m['max_drawdown']:.2%} | {','.join(map(str,item['annual_drawdown_failed_years'])) or '无'} | {item['distinct_traded_etfs']} | {item['trades']} | {'通过' if item['user_historical_target_met'] else '未通过'} |")
        annual.extend({'case':key,**r} for r in item['annual'])
        hashes[path.relative_to(BASE).as_posix()]=digest(path)
    prior=read(directory/'prior_dynamic_reassessment.json')
    primary=results['weekly_group4_risk_parity__base']
    yr=[]
    for r in primary['annual']:
        yr.append(f"| {r['year']}{'（部分年）' if r['partial_year'] else ''} | {r['strategy_return']:.2%} | {r['strategy_max_drawdown']:.2%} | {r['sp500_cny_total_return']:.2%} | {r['sp500_cny_total_max_drawdown']:.2%} |")
    p=directory/'annual_results.csv';pd.DataFrame(annual).to_csv(p,index=False,encoding='utf-8-sig')
    report=f'''# 动态ETF账户策略：新目标首批探索结果

用户修订目标：近十年CAGR不低于人民币SPY含分红的一半；每个共同年份的最大回撤不高于同年基准。当前门槛为{primary['required_cagr']:.2%}。原日均回撤及20%全期最大回撤不再作为本轮用户指定的历史表现硬门槛，仍作补充诊断。

这次研究动态选择ETF及动态权重，固定通信60%/黄金35%不作为解决方案。此前已验收的30个分组/重新分配动态情景按新目标复核，{sum(r['user_historical_target_met'] for r in prior.values())}/30通过。这不代表所有历史研究都已按新口径复核。

首批只改变ETF重选周期：每周为主假设，每月为已验收对照，每日为频率邻域。三者每天检查退出和仓位。保留固定30只候选、63/126/252日正趋势和风险调整排名、行业去重、4槽位风险均衡、15%理想目标预测波动、整手/现金/流动性/换手预算及下一交易日开盘成交。月内退出后较早重选可能减少等待，但也可能追涨杀跌、增加费用，故完整保留所有频率及压力结果。

| 重选频率 | 情景 | CAGR | 全期最大回撤 | 年度回撤未通过年份 | 成交ETF数 | 成交笔数 | 用户历史目标 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
{chr(10).join(rows)}

每周主方案的年度明细：

| 年份 | 策略收益 | 策略最大回撤 | 标普500人民币收益 | 标普500最大回撤 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(yr)}

区间2016-09-12至2026-09-11，2428个中国交易日，从100万元现金开始。年度回撤按年重置高点并纳入上一年末净值；首尾不足整年按共同片段比较。没有在交易中读取当年未来的标普500回撤。

9个情景共独立核对{sum(r['ledger']['days'] for r in results.values()):,}行日账本及{sum(r['signals_checked'] for r in results.values()):,}个信号；特征与独立历史价格前缀逐日核对。三个月度对照的全部成交和净值与旧归档完全相同。所有落盘情景重新核对年度收益连乘和年内高点回撤，基准从原始SPY和汇率重建。周期边界、跨年同周、JSON状态恢复、原目标一致性及退出测试通过。

本批历史目标通过数：{done['passed_cases']}/9。回测通过与实盘准入分开；本轮不更改原每日自动化，不自动下单，不捏造账户或真实前瞻。已反复查看这段历史，新的登记不能将其变为新的样本外数据。

[全部逐年CSV](../{p.relative_to(BASE).as_posix()})。登记和账本目录：`{directory.relative_to(BASE).as_posix()}`。冻结计划见[研究计划](SP500_DYNAMIC_V2_PLAN.md)。
'''
    doc=BASE/'docs/SP500_DYNAMIC_V2_EXPLORATION.md';doc.write_text(report,encoding='utf-8')
    for file in [p,doc,Path(__file__),directory/'registration.json',directory/'results.json',directory/'completion.json',directory/'prior_dynamic_reassessment.json']:
        hashes[file.relative_to(BASE).as_posix()]=digest(file)
    write_json_atomic(directory/'verification.json',{'passed':True,'case_count':9,'performance_passes':done['passed_cases'],
        'input_hashes':reg['input_hashes'],'artifact_hashes':hashes,'daily_dynamic_selection':True,
        'investment_admission':False,'full_goal_achieved':False})
    print(json.dumps({'report':str(doc.relative_to(BASE)),'performance_passes':done['passed_cases'],
        'primary_cagr':primary['strategy']['cagr'],'primary_failed_years':primary['annual_drawdown_failed_years']},ensure_ascii=False))


if __name__=='__main__':run()
