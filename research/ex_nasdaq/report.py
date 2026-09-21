"""从完整验证产物生成报告、曲线和冻结重放配置。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from research.ex_nasdaq.explore import BASE, OUT, write_json
from research.ex_nasdaq.core import SYMBOLS
from research.sp500.report import benchmark_summary


def pct(value):
    return f'{value:.2%}'


def table(items):
    return '| 情景 | 累计收益 | 年化 | 最大回撤 | 收益达标 |\n| --- | ---: | ---: | ---: | --- |\n' + '\n'.join(
        f"| {name} | {pct(v['cumulative_return'])} | {pct(v['cagr'])} | {pct(v['max_drawdown'])} | {'是' if v['target_met'] else '否'} |" for name,v in items.items())


def report():
    directory = Path(json.loads((OUT/'validation/latest.json').read_text(encoding='utf-8'))['directory'])
    frozen = json.loads((directory/'frozen_validation.json').read_text(encoding='utf-8'))
    result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    registry, checks = frozen['registry'], result['checks']
    if not checks['historical_acceptance']:
        raise ValueError('完整历史验收未通过，不生成入选策略')
    selected = {'version': 'ex-nasdaq-1', 'research_only': True, 'historical_goal_verified': True,
        'candidate': frozen['candidate'], 'policy': frozen['policy'], 'source': registry['source'],
        'benchmark': registry['benchmark'], 'starts': registry['starts'], 'end': registry['end'],
        'comparison': registry['comparison'], 'symbols': sorted(SYMBOLS), 'hashes': frozen['hashes'],
        'validation': directory.relative_to(BASE).as_posix()}
    write_json(Path(__file__).with_name('selected_strategy.json'), selected)
    base = pd.read_csv(directory/'report_base_long.csv', index_col=0, parse_dates=True)
    benchmark = pd.read_csv(directory/'benchmark_base_long.csv', index_col=0, parse_dates=True)
    chart = pd.DataFrame({'30只池内入选策略：人民币、扣费': base.account/frozen['policy']['initial_capital'],
                          'SPY全收益：人民币': benchmark.spy_total_cny, 'SPY全收益：美元': benchmark.spy_total_usd})
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2.2,1]})
    for (name,series), color in zip(chart.items(), ['#126958','#bc7a1b','#426aae']):
        axes[0].plot(series.index, series, label=name, color=color, linewidth=1.8)
        axes[1].plot(series.index, (series/series.cummax().clip(lower=1)-1)*100, color=color, linewidth=1.2)
    weights = frozen['candidate']['weights']
    title = f"剔除纳指：{weights['SH515880']:.0%}通信ETF / {weights['SH518880']:.0%}黄金 / 5%现金"
    axes[0].set_title(title, loc='left', fontsize=15, pad=15)
    axes[0].set_ylabel('净值（初始=1）'); axes[1].set_ylabel('回撤（%）')
    axes[0].legend(loc='upper left', frameon=False)
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.text(.07,.02,'2016-01-05至2026-09-11；行业与参数均属已看历史后的开发选择，尚无真实前瞻验收。',fontsize=10,color='#53616b')
    fig.tight_layout(rect=[0,.045,1,1])
    image = BASE/'docs/images/sp500_ex_nasdaq_comparison.png'
    fig.savefig(image,dpi=160,facecolor='white'); fig.savefig(directory/'comparison.svg',facecolor='white')
    plt.close(fig); chart.to_csv(directory/'comparison_wealth.csv')
    benchmarks = {label: benchmark_summary(BASE/registry['benchmark'], v['start'], v['end']) for label,v in result['base'].items()}
    comparisons = []
    for label, v in result['base'].items():
        for name,m in [('策略（扣费）',v),('SPY全收益人民币',benchmarks[label]['spy_total_cny']),
                       ('SPY全收益美元',benchmarks[label]['spy_total_usd']),('标普价格指数美元',benchmarks[label]['sp500_price_usd'])]:
            comparisons.append(f"| {v['start']}起 | {name} | {pct(m['cumulative_return'])} | {pct(m['cagr'])} | {pct(m['max_drawdown'])} |")
    trials = []
    rounds = []
    for run in sorted((OUT/'runs').iterdir()):
        if not run.is_dir() or not (run/'completion.json').exists():
            continue
        results = json.loads((run/'results.json').read_text(encoding='utf-8'))
        rounds.append({'run': run.name, 'candidates': len(results)})
        for name, metrics in results.items():
            a,b = metrics['long'],metrics['recent']
            trials.append(f"| {run.name} | {name} | {pct(a['cagr'])} | {pct(a['max_drawdown'])} | {pct(b['cagr'])} | {'是' if a['target_met'] and b['target_met'] else '否'} |")
    diag = result['diagnostics']; ci = diag['excess_daily_block_ci']
    yearly = '\n'.join(f"| {year} | {pct(v)} | {pct(benchmarks['base_long']['spy_total_cny']['annual_returns'][year])} |" for year,v in result['base']['base_long']['annual_returns'].items())
    content = f'''# 剔除纳指后的30只ETF：标普500对标研究验收

日期：2026-09-13。验证运行：`{directory.name}`。

**历史研究验收通过；实盘投资准入未通过。** 入选规则：{title}。通信ETF（515880）不符合当日资格时，以合格沪深300ETF（510300）承接权益目标；均不合格则现金。黄金ETF为518880。交易、信号和行情均严格排除513100，实际操作3只，均属于剩余30只研究池。

## 对标结果

| 区间（截至2026-09-11） | 对象 | 累计收益 | 年化 | 最大回撤 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(comparisons)}

预先固定的收益门槛：年化落后人民币SPY不超过2个百分点，累计收益至少达到其85%；两个起点均以100万元现金重新建仓，扣除交易成本，不使用杠杆或做空。美国收盘只用于事后基准比较，不进入中国当日信号。表中基准回撤采用全部美国交易日；曲线向后对齐中国交易日，汇率与分红口径沿用已核对的冻结快照。

![净值与回撤](images/sp500_ex_nasdaq_comparison.png)

## 固定执行规则

1. 上市观察、20日日均成交额至少5000万元、当时已公告规模至少2亿元、缺价及退市检查保留。通信的资格缺位按预先规定的沪深300代理处理；没有为历史早期补入当时不合格的行业ETF。
2. 每日收盘后按固定权重检查漂移，次一中国交易日开盘执行。再平衡偏差带3个百分点；保留原权重后再次核对风险；资格退出优先。
3. 观察波动，不按预测波动强制缩仓；目标总仓位不超过95%，单只与同组下单上限85%。日双边换手预算30%，买入仅用卖出需求扣除后的剩余额度与实际现金；价格漂移可能使盘后权重偏离目标，下一决策再调整。
4. 买入按100份整数倍；参与率上限为20日日均成交额的2%。佣金单边0.03%，最低0.2元；基础滑点5bp，加参与率冲击；现金不计息。缺价、停牌与价格限制拒绝成交，卖单未成交时不预支现金。
5. 研究回撤预算30%，原投资准入20%线另报。这些历史预算均不保证未来最大亏损。

本方案存在较高的单一通信行业暴露，不能视为对标普500的日常跟踪。长期早期收益部分来自沪深300代理，近年结果依赖通信与黄金的历史行情。原正式策略及默认账户未切换。

## 成交压力

{table(result['stress'])}

6个压力均为正收益且回撤不超过30%：{checks['stress_positive_within_budget']}。每个情景的收益是否仍接近标普已单列，不将“压力下仍盈利”写成“压力下均达到收益门槛”。资本、整数份额和调仓路径会使摩擦与最终收益不呈严格单调关系。

## 权重邻域与附加检查

通信50%/55%/60%/65%/70%，黄金补足95%总目标；{checks['neighbors_passing']}/{checks['neighbor_count']}组同时满足长期收益目标与30%回撤预算，预登记门槛为至少2/3。主候选事前固定为60%，没有选择收益最大的权重点。

{table(result['neighbors'])}

{table(result['additional'])}

2016–2020分段诊断：年化{pct(diag['early_2016_2020']['cagr'])}，最大回撤{pct(diag['early_2016_2020']['max_drawdown'])}。该段和滚动窗口都属于已看历史，不是新的独立样本外确认。

## 年度表现与统计限制

| 年度（2026截至9月11日） | 策略 | SPY全收益人民币 |
| --- | ---: | ---: |
{yearly}

最差滚动252交易日收益差为{pct(diag['worst_12month_gap'])}，最好为{pct(diag['best_12month_gap'])}；月收益相关系数{diag['monthly_correlation']:.3f}。日均超额收益21日块bootstrap的95%区间为[{pct(ci['ci_low'])}, {pct(ci['ci_high'])}]，未经多重选择调整，不能据此声称显著alpha。

本轮{len(rounds)}批、{len(trials)}个登记配置回放（含重复对照），并保留此前31只池的34个已查看候选。通信来自全行业历史筛选，权重邻域通过不能消除行业选择偏差。当前存续池仍缺完整历史目录与公司行动审计，真实前瞻0/126日，投资准入不成立。

## 工程证据与重放

- 两个起点的独立验证净值差额：{checks['reproduction_error_cny']}元；全部完整情景完成逐日现金/份额账本核对，信号早于成交、买入资格、整手和参与率检查。
- 所有成交属于30只池：{checks['only_30_symbols_traded']}；生产指针保持不变：{checks['production_pointer_unchanged']}；基础两个区间均满足原20%回撤线：{checks['original_20_percent_drawdown_passed']}。
- 本轮相关工程检查见`output/sp500_ex_nasdaq/verification/unittest.txt`，最终重放、哈希与产物验收以`output/sp500_ex_nasdaq/completion_verification.json`为准。

```powershell
.venv\\Scripts\\python.exe -m research.ex_nasdaq.replay
.venv\\Scripts\\python.exe -m research.ex_nasdaq.validate --protocol research/ex_nasdaq/validation_protocol.json
```

规则：[selected_strategy.json](../research/ex_nasdaq/selected_strategy.json)。验证：[results.json](../{directory.relative_to(BASE).as_posix()}/results.json)。研究计划：[SP500_EX_NASDAQ_PLAN.md](SP500_EX_NASDAQ_PLAN.md)。

## 全部候选（包含失败结果）

| 批次 | 候选 | 长期年化 | 长期回撤 | 近期年化 | 双区间收益达标 |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(trials)}
'''
    (BASE/'docs/SP500_EX_NASDAQ_ACCEPTANCE.md').write_text(content,encoding='utf-8')
    write_json(directory/'report_supplement.json', {'benchmarks': benchmarks, 'rounds': rounds, 'trial_count': len(trials)})
    print(json.dumps({'report': 'docs/SP500_EX_NASDAQ_ACCEPTANCE.md', 'trials': len(trials), 'historical_acceptance': True},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    report()
