"""生成全部资金重新分配情景的配对报告与静态比较图。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.audit_redeployment_outcome import benchmark_from_raw
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.run_redeployment import OUT, PRIOR
from research.goal30.turnover_budget import require

NAMES = {'group4_risk_parity': '分组4风险均衡（主方案）', 'ungrouped4_equal': '无分组4等权',
    'group4_equal': '分组4等权', 'group3_risk_parity': '分组3风险均衡', 'group5_risk_parity': '分组5风险均衡'}
SCENARIOS = {'base': '基础', 'slippage_2': '滑点2倍', 'payment_delay_5': '支付延迟5日'}


def report(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT), '需使用本研究登记目录')
    audit = read(directory/'outcome_audit.json')
    result = read(directory/'results.json'); proof = read(directory/'verification.json')
    require(audit['passed'] and proof['passed'], '独立验收未通过')
    for rel, expected in audit['input_and_artifact_hashes'].items():
        require(digest(BASE/rel) == expected, '报告输入变化：'+rel)
    benchmark, _ = benchmark_from_raw()
    b = audit['benchmark_from_raw_spy_and_fx']; primary = audit['cases']['group4_risk_parity__base']
    sys.path.append('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    matplotlib.rcParams['font.family'] = font.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    curves = [(pd.read_csv(PRIOR/'group4_risk_parity__base_report.csv', index_col='date', parse_dates=True).account,
               '配对旧主方案', '#8292a5'),
              (pd.read_csv(directory/'group4_risk_parity__base_report.csv', index_col='date', parse_dates=True).account,
               '重新分配主方案', '#197e80'), (benchmark, '人民币SPY含分红', '#bb693d')]
    for curve, name, color in curves:
        axes[0, 0].plot(curve.index, curve/curve.iloc[0], label=name, color=color, lw=1.6)
        axes[0, 1].plot(curve.index, 1-curve/curve.cummax(), label=name, color=color, lw=1.1)
    axes[0, 0].set_title('主方案：十年净值（起点 = 1）', pad=12)
    axes[0, 0].set_ylabel('归一化净值'); axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_title('主方案：每日回撤幅度（越小越好）', pad=12)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    keys = [name+'__base' for name in NAMES]
    labels = [NAMES[name].replace('（主方案）', '\n主方案').replace('风险均衡', '\n风险均衡') for name in NAMES]
    x = np.arange(len(keys)); width = .34
    for ax, metric, title in [(axes[1, 0], 'cagr', '五种基础方案：年化收益'),
                              (axes[1, 1], 'mean_daily_drawdown', '五种基础方案：日均回撤')]:
        for offset, version, name, color in [(-width/2, 'prior', '配对旧版本', '#8292a5'),
                                            (width/2, 'new', '重新分配版本', '#197e80')]:
            values = [audit['cases'][k][version]['metrics'][metric] for k in keys]
            bars = ax.bar(x+offset, values, width=width, color=color, label=name)
            ax.bar_label(bars, labels=[f'{v:.2%}' for v in values], padding=3, fontsize=8)
        ax.axhline(b[metric], color='#bb693d', ls='--', lw=1.2, label=f'人民币SPY {b[metric]:.2%}')
        ax.set_xticks(x, labels, fontsize=9); ax.set_title(title, pad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, max(b[metric], max(audit['cases'][k][v]['metrics'][metric] for k in keys for v in ('new', 'prior')))*1.22)
        ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('资金重新分配研究：提高仓位未能满足收益与回撤联合目标', fontsize=18, y=.97)
    fig.text(.5, .025, '2016-09-12 — 2026-09-11 · 2428个共同交易日 · 策略扣除成本 · 历史研究，未获投资准入',
             ha='center', fontsize=10, color='#555')
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.13, hspace=.43, wspace=.19)
    figure = directory/'paired_comparison.png'; fig.savefig(figure, dpi=160); plt.close(fig)
    full_rows = []; paired_rows = []; early_rows = []; cost_rows = []; flat = []
    for key, row in audit['cases'].items():
        variant, scenario = key.split('__'); new, old = row['new'], row['prior']; m = new['metrics']; d = row['paired']
        full_rows.append(f"| {NAMES[variant]} | {SCENARIOS[scenario]} | {m['cagr']:.2%} | {m['mean_daily_drawdown']:.2%} | {m['max_drawdown']:.2%} | {m['mean_annual_max_drawdown']:.2%} | {'通过' if new['checks']['joint_admission_performance'] else '未通过'} |")
        flat.append({'case': key, **{'new_'+k:v for k,v in new.items() if not isinstance(v, dict)},
                     **{'prior_'+k:v for k,v in old.items() if not isinstance(v, dict)}, **d, **row['paired_cost'],
                     **{'new_'+k:v for k,v in m.items() if not isinstance(v, dict)},
                     **{'prior_'+k:v for k,v in old['metrics'].items() if not isinstance(v, dict)},
                     **row['signals']})
        if scenario != 'base': continue
        paired_rows.append(f"| {NAMES[variant]} | {old['mean_invested_fraction']:.2%} → {new['mean_invested_fraction']:.2%} | {d['cagr_delta']*100:+.2f} | {d['mean_daily_drawdown_delta']*100:+.2f} | {d['max_drawdown_delta']*100:+.2f} | {d['trade_count_delta']:+d} | {old['average_annual_two_way_turnover']:.2f} → {new['average_annual_two_way_turnover']:.2f} |")
        early_rows.append(f"| {NAMES[variant]} | {old['pre_2025_metrics']['cagr']:.2%} | {new['pre_2025_metrics']['cagr']:.2%} | {new['pre_2025_metrics']['mean_daily_drawdown']:.2%} |")
        cost_rows.append(f"| {NAMES[variant]} | {old['total_commission_cny']:,.0f} → {new['total_commission_cny']:,.0f} | {old['total_slippage_and_impact_cny']:,.0f} → {new['total_slippage_and_impact_cny']:,.0f} |")
    csv = directory/'paired_comparison.csv'; pd.DataFrame(flat).to_csv(csv, index=False, encoding='utf-8-sig')
    annual_rows = []
    for year in result['cases']['group4_risk_parity__base']['annual_returns']:
        values = [result['cases'][key]['annual_returns'][year]['strategy'] for key in keys]
        bm = result['cases'][keys[0]]['annual_returns'][year]['spy_total_cny']
        annual_rows.append('| '+year+('（不足整年）' if year in ('2016', '2026') else '')+' | '+
                           ' | '.join(f'{v:.2%}' for v in [*values, bm])+' |')
    p, delta = primary['new'], primary['paired']
    pre = result['cases']['group4_risk_parity__base']['pre_2025_diagnostic']['spy_total_cny']
    max_risk = max(row['signals']['maximum_post_budget_target_forecast_volatility'] for row in audit['cases'].values())
    breaches = proof['post_budget_target_risk_exceedance_case_days']
    report_text = f'''# 原风险约束内资金重新分配：验收结果

日期：2026-09-14。**本阶段实现与账本核验通过，15个完整历史情景的投资表现联合门槛通过数为{audit['full_history_joint_performance_passes']}/15，仍未达到投资准入。** 主方案平均仓位提高{delta['mean_invested_fraction_delta']*100:.2f}个百分点，CAGR升至{p['metrics']['cagr']:.2%}，但日均回撤为{p['metrics']['mean_daily_drawdown']:.2%}；人民币SPY分别为{b['cagr']:.2%}和{b['mean_daily_drawdown']:.2%}。

## 比较范围与口径

预登记目录：`{directory.relative_to(BASE).as_posix()}`，1280项输入指纹。保留固定30只候选、每天收盘检查、次日开盘按条件执行。五种原方案各运行基础、滑点2倍、支付延迟5交易日，共15个完整十年情景。配对旧版本使用上一轮已核验的新分红支付模型，见[原件与支付模型验收](SP500_REMAINING_DIVIDENDS_ACCEPTANCE.md)。

只改变第一次仓位限制：按原风险组及原权重比例，在单只35%、组别70%、总仓位95%以内重新分配被截断资金，再同比例缩至原15%预测波动。预算不超过原始目标总和，容量不足仍留现金。原月初选择、月内退出后等待下月、趋势、资格、缺价要求、3个百分点漂移带、第二次限制与每日30%共享换手预算均保留。选择规则相同不保证实际每日选中标的相同，成交与退出状态会反馈到后续选择。

2016-09-12—2026-09-11，共2428个中国交易日，100万元现金起步，费用计入净值。基准为同交易日的人民币SPY含分红。日均回撤是各交易日相对历史净值高点的损失均值，新高日计零；另报年度最大回撤均值。此处将平均回撤明确定义以便复核，并不声称业内只有一种口径。主方案全部压力及3/4/5槽位中至少两个基础邻域需通过CAGR、日均回撤及20%最大回撤联合要求，才进入后续准入审查。

![十年净值、回撤及五方案配对比较](../{figure.relative_to(BASE).as_posix()})

## 全部15个情景

| 方法 | 情景 | CAGR | 日均回撤 | 最大回撤 | 年度最大回撤均值 | 联合门槛 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(full_rows)}
| 人民币SPY含分红 | 共同基准 | {b['cagr']:.2%} | {b['mean_daily_drawdown']:.2%} | {b['max_drawdown']:.2%} | {b['mean_annual_max_drawdown']:.2%} | 策略比较基准 |

## 基础情景配对变化

收益与回撤变化列单位为百分点，回撤变大表示风险恶化。平均仓位为每日ETF市值/经济净值，分母含分红应收；年均双边换手为每日买卖原价金额/前日净值之和，再除以实际年数，单位为倍。

| 方法 | 平均仓位（旧→新） | CAGR变化 | 日均回撤变化 | 最大回撤变化 | 成交笔数变化 | 年均双边换手（旧→新） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(paired_rows)}

| 方法 | 十年佣金（旧→新，元） | 十年滑点与冲击成本（旧→新，元） |
| --- | ---: | ---: |
{chr(10).join(cost_rows)}

成本为对应路径实际成交模型的名义金额合计，不能直接当作CAGR损失。完整15情景的配对数值、成本、仓位与预测风险见[比较CSV](../{csv.relative_to(BASE).as_posix()})。三槽位风险均衡在双倍滑点下最大回撤超过20%；五槽位基础方案增加仓位后CAGR下降、日均回撤扩大。资金利用率改善的收益效果不稳定，不据此推广为默认策略。

## 逐年收益与早期诊断

| 年份 | 分组4风险均衡 | 无分组4等权 | 分组4等权 | 分组3风险均衡 | 分组5风险均衡 | 人民币SPY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(annual_rows)}

以下截至2024-12-31，仍是已查看历史的诊断区间，不是新增样本外测试。共同基准CAGR为{pre['cagr']:.2%}、日均回撤{pre['mean_daily_drawdown']:.2%}。

| 方法 | 旧CAGR | 新CAGR | 新日均回撤 |
| --- | ---: | ---: | ---: |
{chr(10).join(early_rows)}

## 实现和独立验收

六组测试通过：手算分配、容量不足、非法输入、组约束和预算、100组随机分配与独立二分/顺序对照、高波动缩仓、月内状态恢复及未来价格扰动。回放完成后，从落盘JSON独立重建{proof['ledger_rows']:,}个日账本行和{proof['signals_verified']:,}个信号，核查权益、现金、整手、费用、参与率、每日预算及账户恢复。账本最大数值差为{proof['maximum_ledger_error']:.3g}。

分配复核使用独立二分法，风险复核使用直接矩阵协方差。换手预算后的目标预测波动最高为{max_risk:.6%}，超过15%的情景日为{breaches}（容差1e-10）。这是目标权重的历史诊断；实际成交、价格波动与退出等待会导致持仓偏离，不能将15%描述成实际持仓的硬风险上限。

另从原始SPY、汇率和新旧账本独立核验{audit['paired_report_rows']:,}行，复算CAGR、日均及年度回撤、全期最大回撤、逐年收益、2025年前诊断、平均仓位、交易笔数、成本与配对差额。原始基准和CSV/JSON一致。

复核命令：

```powershell
.venv\\Scripts\\python.exe -m research.goal30.run_redeployment verify --directory {directory.relative_to(BASE).as_posix()}
.venv\\Scripts\\python.exe -m research.goal30.audit_redeployment_outcome {directory.relative_to(BASE).as_posix()}
```

## 研究决策与尚缺条件

本机制保留为研究分支，不提升为每日执行版本。更高仓位不足以弥补原选择与时点形成的收益差距；既有[收益归因](SP500_GROUP_RETURN_ATTRIBUTION.md)仍显示初始历史窗口、月内移除后等待与持仓相对收益需要分别处理。

下一步先针对已有的8个上市后缺价标的日查验真实行情及停牌/拆分原件，分清合理无交易与数据缺漏，再研究初始历史窗口的真实数据扩展；不插值、不补造上市前价格，也不同时修改信号与重入时点。该数据工作用于修复输入证据，不能预先称为策略收益改进。本批结果不支持继续按收益高低挑选仓位参数。

当前仍缺满足联合表现的策略、完整历史行动及存续/清盘目录、账户及真实到账核对、126日真实前瞻（现为0/126）。同一段历史被反复研究的事实保留；预登记新算法不会产生独立样本外证据。正式指针仍为`20260912-220345-f20ea3f5`，每日冻结候选及自动化未切换，本轮未生成实盘订单。
'''
    doc = BASE/'docs/SP500_REDEPLOYMENT_ACCEPTANCE.md'; doc.write_text(report_text, encoding='utf-8')
    source_files = [directory/'outcome_audit.json', directory/'verification.json', directory/'results.json']
    output_files = [doc, figure, csv, Path(__file__)]
    write_json_atomic(directory/'report_manifest.json', {'source_hashes': {p.relative_to(BASE).as_posix(): digest(p) for p in source_files},
        'report_hashes': {p.relative_to(BASE).as_posix(): digest(p) for p in output_files}, 'figure_visually_reviewed': False})
    print(json.dumps({'report': doc.relative_to(BASE).as_posix(), 'figure': figure.relative_to(BASE).as_posix()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)
