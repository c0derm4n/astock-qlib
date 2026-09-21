"""从已独立验收的完整结果生成中文报告与收益/回撤图。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as mtick
import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import REFERENCE

LABELS = {'group4_risk_parity': '分组4槽位风险均衡（主方案）', 'ungrouped4_equal': '无分组4槽位等权',
    'group4_equal': '分组4槽位等权', 'group3_risk_parity': '分组3槽位风险均衡', 'group5_risk_parity': '分组5槽位风险均衡'}
SHORT = {'group4_risk_parity': '主方案4槽位', 'ungrouped4_equal': '无分组等权',
    'group4_equal': '分组等权', 'group3_risk_parity': '3槽位邻域', 'group5_risk_parity': '5槽位邻域'}
SCENARIO_NAMES = {'base': '基础', 'slippage_2': '滑点2倍', 'payment_delay_5': '支付延迟5日'}


def report(directory):
    directory = Path(directory).resolve()
    audited = read(directory/'outcome_audit.json')
    result = read(directory/'results.json')
    proof = read(directory/'verification.json')
    registered = read(directory/'registration.json')
    require(audited['results_sha256'] == digest(directory/'results.json') and
            audited['verification_sha256'] == digest(directory/'verification.json'), '报告前结果或复核被改变')
    main_key = 'group4_risk_parity__base'
    main = result['cases'][main_key]
    benchmark = audited['benchmark_from_raw_spy_and_fx']
    require(len(result['cases']) == 15 and all(not c['metrics']['checks']['daily_definition_target_met']
            for c in result['cases'].values()), '本版失败机制报告只适用于全部15情景未达目标的已验收结果')
    primary = BASE/registered['primary_execution_archive']
    control = read(primary/'results.json')['cases']['fixed_control__base']['metrics']['strategy']
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), gridspec_kw={'width_ratios': [1.18, 1]})
    fig.set_facecolor('#f8fafc')
    main_report = pd.read_csv(directory/(main_key+'_report.csv'), index_col='date', parse_dates=True).account
    fixed_report = pd.read_csv(primary/'fixed_control__base_report.csv', index_col='date', parse_dates=True).account
    spy = pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv', index_col=0, parse_dates=True).spy_total_cny
    for series, label, color in [(spy, '人民币SPY含分红', '#64748b'), (fixed_report, '原固定候选', '#ea580c'),
                                  (main_report, '分组风险均衡主方案', '#0369a1')]:
        axes[0].plot(series.index, series/series.iloc[0]-1, label=label, color=color, lw=1.9)
    axes[0].set_title('十年累计收益：策略扣除交易成本', loc='left', pad=15)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axes[0].legend(frameon=False, loc='upper left')
    axes[0].grid(alpha=.15)
    x, y = benchmark['mean_daily_drawdown']*100, benchmark['cagr']*100
    axes[1].fill_between([0, x], y, 19, color='#dcfce7', alpha=.8)
    axes[1].axvline(x, color='#94a3b8', ls='--', lw=1)
    axes[1].axhline(y, color='#94a3b8', ls='--', lw=1)
    axes[1].text(.3, 18, '收益与日均回撤目标区域', fontsize=9, color='#166534')
    for i, name in enumerate(LABELS):
        row = audited['cases'][name+'__base']
        px, py = row['daily_drawdown']*100, row['cagr']*100
        axes[1].scatter(px, py, s=45, color='#0369a1' if i==0 else '#7c3aed')
        label_positions = [(6.1, 3.1), (8.5, 7.5), (6.4, 8.0), (8.6, 3.1), (3.8, 5.2)]
        axes[1].annotate(SHORT[name], (px, py), xytext=label_positions[i], textcoords='data', fontsize=8,
                         arrowprops={'arrowstyle': '-', 'color': '#94a3b8', 'lw': .8})
    axes[1].scatter(control['mean_daily_drawdown']*100, control['cagr']*100, color='#ea580c', s=55)
    axes[1].annotate('原固定候选', (control['mean_daily_drawdown']*100, control['cagr']*100),
                     xytext=(8,5), textcoords='offset points', fontsize=9)
    axes[1].scatter(x,y, marker='D', s=55, color='#64748b')
    axes[1].annotate('人民币SPY', (x,y), xytext=(-12,-18), textcoords='offset points', fontsize=9)
    axes[1].set(xlim=(0,11), ylim=(0,19), xlabel='逐日平均回撤（%，越低越好）', ylabel='CAGR（%）')
    axes[1].set_title('五种基础方案均未进入目标区域', loc='left', pad=15)
    axes[1].grid(alpha=.15)
    fig.suptitle('固定30只ETF · 分组与风险分散研究', x=.065, ha='left', fontsize=16, weight='bold')
    fig.text(.065, .02, '2016-09-12—2026-09-11；2428个共同中国交易日。已看历史的开发研究，投资准入未通过。', fontsize=9, color='#475569')
    fig.tight_layout(rect=(.01,.05,1,.93), w_pad=3)
    figure = directory/'group_comparison.png'
    fig.savefig(figure, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    rows = ['# 分组趋势与风险分散研究验收', '',
        '日期：2026-09-14。5种方案、15个预登记基础及压力情景全部完成，独立账本、信号和指标验收通过；**0/15达到收益与逐日平均回撤联合目标，不采纳为最终策略。**', '',
        f"主方案基础年化{main['metrics']['strategy']['cagr']:.2%}、日均回撤{main['metrics']['strategy']['mean_daily_drawdown']:.2%}、最大回撤{main['metrics']['strategy']['max_drawdown']:.2%}。虽通过20%最大回撤门槛，但收益远低于人民币SPY的{benchmark['cagr']:.2%}，日均回撤也高于{benchmark['mean_daily_drawdown']:.2%}。工程通过不改变投资目标失败的结论。", '',
        '## 统一口径与完整结果', '',
        '2016-09-12—2026-09-11，共2428个中国交易日；每个情景独立从100万元现金开始。策略扣除佣金、滑点及冲击，SPY含分红折算人民币。日均回撤为相对经济净值历史高点的每日损失均值，新高日计零。年度最大回撤均值保留为补充。', '',
        '| 方案 | 情景 | CAGR | 日均回撤 | 最大回撤 |', '| --- | --- | ---: | ---: | ---: |']
    for name, label in LABELS.items():
        for scenario, display in SCENARIO_NAMES.items():
            r = audited['cases'][name+'__'+scenario]
            rows.append(f"| {label} | {display} | {r['cagr']:.2%} | {r['daily_drawdown']:.2%} | {r['max_drawdown']:.2%} |")
    rows += [f"| 原固定候选 | 基础控制 | {control['cagr']:.2%} | {control['mean_daily_drawdown']:.2%} | {control['max_drawdown']:.2%} |",
             f"| 人民币SPY含分红 | 共同基准 | {benchmark['cagr']:.2%} | {benchmark['mean_daily_drawdown']:.2%} | {benchmark['max_drawdown']:.2%} |", '',
        f"![累计收益与联合目标区域](../{figure.relative_to(BASE).as_posix()})", '', '## 机制判断', '',
        '这批方法从全池选择趋势代表，按固定产业/资产组避免重复持仓，再以协方差分配风险。对照拆分了分组和风险均衡的作用；3/5槽位邻域在看结果前固定。五种方案均无法保持足够收益，因此没有证据支持用这些规则替换原固定候选。', '',
        f"主方案平均投资仓位{main['mean_invested_fraction']:.2%}，十年{main['trade_count']}笔成交；日双边换手率累加{main['sum_daily_two_way_turnover']:.2f}，年均约{audited['cases'][main_key]['average_annual_two_way_turnover']:.2f}倍。滑点加倍后年化进一步下降。", '',
        f"253条连续真实价格要求与月度复查使首次成交出现在{audited['cases'][main_key]['first_trade']}；此前现金等待保留在十年结果中。停牌/缺价可能使信号连续性要求暂时不满足，未用插值或上市前数据恢复资格。", '',
        f"截至2024年底，主方案同起点年化仅{main['pre_2025_diagnostic']['strategy']['cagr']:.2%}，同期SPY为{main['pre_2025_diagnostic']['spy_total_cny']['cagr']:.2%}。2025年虽上涨{main['annual_returns']['2025']['strategy']:.2%}，仍不足以达到十年要求；不删去此前亏损年。", '',
        '## 主方案逐年收益', '', '| 年份 | 主方案 | 人民币SPY |', '| --- | ---: | ---: |']
    for year, value in main['annual_returns'].items():
        suffix = '（部分年度）' if value['partial_year'] else ''
        rows.append(f"| {year}{suffix} | {value['strategy']:.2%} | {value['spy_total_cny']:.2%} |")
    rows += ['', '## 验收证据', '',
        f"- 5组新增规则测试通过，含未来价格扰动、复权尺度、独立前缀、缺价、同分、组内选择、风险贡献收敛和月内退出状态恢复。", 
        f"- {proof['input_hashes_verified']}项冻结输入及3份登记源码副本一致；15个情景完整保留。原数据、旧代码、正式指针及每日自动化未替换。",
        f"- 每个信号日检查全部30只，独立核对{proof['feature_verification']['valid_symbol_scores']}个有效标的分数。15个情景共重建{proof['signals_reconstructed']:,}条账户相关信号与{proof['daily_rows_reconstructed']:,}个日账本行，逐笔确认资格、整手、参与率、费用和30%共享买卖预算。",
        f"- 独立账本最大数值误差{proof['maximum_numeric_error']:.3g}元。另从原SPY及USD/CNY重建基准，重算全部情景的CAGR、两类平均回撤、最大回撤、全部年度收益及2025年前诊断；15份报告CSV与JSON一致。",
        '- 主方案全部压力未通过，3/4/5槽位基础邻域0/3通过，预登记准入研究门槛失败。这些相同历史情景不是独立样本，也不计126日真实前瞻。', '',
        '## 复核命令与未完成项', '', f'运行目录：`{directory.relative_to(BASE).as_posix()}`。', '', '```powershell',
        f'.venv\\Scripts\\python.exe -m research.goal30.run_group_diversification --verify {directory.relative_to(BASE).as_posix()}',
        f'.venv\\Scripts\\python.exe -m research.goal30.audit_group_outcome {directory.relative_to(BASE).as_posix()}', '```', '',
        '仍未找到满足原目标的策略。下一步需补齐剩余31项区间分红原件与两项预热证据、全池完整目录，并据失败归因评估新的收益机制。用户账户及126日真实前瞻仍缺，不提供实盘买卖份额。不能仅在本批失败参数附近继续挑选，或放宽收益/回撤目标宣称通过。']
    path = BASE/'docs/SP500_GROUP_DIVERSIFICATION_ACCEPTANCE.md'
    path.write_text('\n'.join(rows)+'\n', encoding='utf-8')
    write_json_atomic(directory/'report_manifest.json', {'inputs': {p.relative_to(BASE).as_posix():digest(p) for p in
        (directory/'outcome_audit.json', directory/'verification.json', directory/'results.json', Path(__file__))},
        'artifacts': {p.relative_to(BASE).as_posix():digest(p) for p in (path, figure)}, 'visual_review_completed': False})
    print(path); print(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)
