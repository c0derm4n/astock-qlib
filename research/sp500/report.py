"""从机器产物生成中文验收报告、标准净值图和可校验重放配置。"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from research.sp500.explore import BASE
from research.sp500.benchmark import aligned_wealth
from src.simulation import performance


def pct(value):
    return f"{value:.2%}"


def benchmark_summary(directory, start, end):
    dates = pd.read_csv(directory/"spy.csv", index_col="date", parse_dates=True).loc[start:end].index
    wealth = aligned_wealth(directory, dates)
    return {name: performance(pd.DataFrame({"return": values.pct_change(fill_method=None).fillna(0),
                                           "cost": 0., "turnover": 0.})) for name,values in wealth.items()}


def make_report():
    root = BASE/"output/sp500_exploration"
    directory = Path(json.loads((root/"validation/latest.json").read_text(encoding="utf-8"))["directory"])
    result = json.loads((directory/"results.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory/"frozen_validation.json").read_text(encoding="utf-8"))
    source, benchmark_dir = Path(frozen["source"]), Path(frozen["benchmark"])
    registry = json.loads((root/"runs"/frozen["protocol"]["exploration_run"]/"registry.json").read_text(encoding="utf-8"))
    primary, recent = result["base"]["primary"], result["additional"]["restart_2021"]
    base = pd.read_csv(directory/"report_primary.csv", index_col=0, parse_dates=True)
    bm_long = benchmark_summary(benchmark_dir, primary["start"], primary["end"])
    bm_recent = benchmark_summary(benchmark_dir, recent["start"], recent["end"])
    benchmark = aligned_wealth(benchmark_dir, base.index)
    checks = result["checks"]
    count, passed = checks["neighbor_count"], checks["neighbors_target_within_research_budget"]
    completed = bool(primary["target_met"] and recent["target_met"] and
                     checks["all_execution_stress_positive_within_research_budget"] and passed >= math.ceil(count*2/3)
                     and checks["production_pointer_unchanged"] and not checks["production_code_mismatches"])
    chart = pd.DataFrame({"入选策略：人民币，扣费": base.account/frozen["execution_policy"]["initial_capital"],
                          "SPY全收益：折算人民币": benchmark.spy_total_cny,
                          "SPY全收益：美元": benchmark.spy_total_usd})
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for (label, series), color in zip(chart.items(), ["#126958", "#bd7718", "#426aae"]):
        axes[0].plot(series.index, series, label=label, color=color, linewidth=1.8)
        axes[1].plot(series.index, (series/series.cummax().clip(lower=1)-1)*100, color=color, linewidth=1.3)
    axes[0].set_title("固定31只ETF内的配置：65%海外权益 / 30%黄金 / 5%现金", loc="left", fontsize=15, pad=16)
    axes[0].set_ylabel("净值（初始=1）")
    axes[1].set_ylabel("回撤（%）")
    axes[0].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.07, .02, f"{primary['start']}至{primary['end']}；34个含基线的开发候选。海外收盘仅用于事后比较，未进入中国当日信号。", fontsize=10, color="#535d68")
    fig.tight_layout(rect=[0, .045, 1, 1])
    image = BASE/"docs/images/sp500_strategy_comparison.png"
    fig.savefig(image, dpi=160, facecolor="white")
    fig.savefig(directory/"comparison.svg", facecolor="white")
    plt.close(fig)
    chart.to_csv(directory/"comparison_wealth.csv")
    original = json.loads((source/"run_meta.json").read_text(encoding="utf-8"))
    required = dict(original["code"]["files"])
    for p in ([source/name for name in ("market_snapshot.parquet", "nav_snapshot.parquet", "metadata_snapshot.parquet")]
              + list(Path(__file__).parent.glob("*.py")) + list(benchmark_dir.glob("*.csv"))):
        required[p.relative_to(BASE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    keys = ("initial_capital", "max_weight", "max_group_weight", "max_equity_weight", "max_gross_weight", "vol_target",
            "volatility_policy", "covariance_days", "max_daily_turnover", "max_adv_participation", "commission",
            "minimum_commission", "slippage", "impact_coefficient", "rebalance_band", "annualization_days", "conservative_buy_budget")
    policy = {k: frozen["execution_policy"][k] for k in keys if k in frozen["execution_policy"]}
    selected = {"version": "sp500-candidate-3", "research_only": True, "historical_goal_verified": completed,
                "candidate": frozen["candidate"], "policy": policy, "start": registry["start"], "end": registry["end"],
                "comparison": registry["comparison"], "source_directory": source.relative_to(BASE).as_posix(),
                "benchmark_directory": benchmark_dir.relative_to(BASE).as_posix(),
                "validation_directory": directory.relative_to(BASE).as_posix(), "required_hashes": required}
    Path(__file__).with_name("selected_strategy.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    supplement = {"benchmark_long_us_calendar": bm_long, "benchmark_recent_us_calendar": bm_recent,
                  "historical_goal_verified": completed}
    (directory/"supplement.json").write_text(json.dumps(supplement, ensure_ascii=False, indent=2), encoding="utf-8")
    header = "| 名称 | 累计收益 | CAGR | 最大回撤 | 接近同期人民币SPY |\n| --- | ---: | ---: | ---: | --- |\n"
    table = lambda items: header + "\n".join(f"| {name} | {pct(m['cumulative_return'])} | {pct(m['cagr'])} | {pct(m['max_drawdown'])} | {'是' if m['target_met'] else '否'} |" for name,m in items.items())
    comparison = "\n".join(f"| {period} | {label} | {pct(m['cumulative_return'])} | {pct(m['cagr'])} | {pct(m['max_drawdown'])} |" for period,strategy,bm in [
        (f"{primary['start']}—2026-09-11", primary, bm_long), ("2021-01-04—2026-09-11（重新建仓）", recent, bm_recent)]
        for label,m in [("策略：人民币、扣费",strategy),("SPY全收益：人民币",bm['spy_total_cny']),("SPY全收益：美元",bm['spy_total_usd']),("标普价格指数：美元",bm['sp500_price_usd'])])
    yearly = "\n".join(f"| {year}{'（截至9/11）' if year=='2026' else ''} | {pct(value)} | {pct(bm_long['spy_total_cny']['annual_returns'][year])} |" for year,value in primary["annual_returns"].items())
    all_trials = []
    for run in ("20260913-080733", "20260913-081152", "20260913-083541", "20260913-084158", "20260913-085603", "20260913-085931"):
        payload = json.loads((root/"runs"/run/"results.json").read_text(encoding="utf-8"))
        for name,m in payload["candidates"].items():
            all_trials.append(f"| {name} | {m['start']} | {pct(m['cumulative_return'])} | {pct(m['cagr'])} | {pct(m['max_drawdown'])} | {'是' if m['historical_target_met'] else '否'} |")
    diag, ci = result["diagnostics"], result["diagnostics"]["excess_daily_block_ci"]
    report = f'''# 31只ETF对标标普500：探索与验收

日期：2026-09-13。最终验证：`{directory.name}`。

**历史探索验收：{'通过' if completed else '尚未通过全部要求'}。** 固定65%纳指ETF（513100）、30%黄金ETF（518880）、至少5%现金；纳指不符合当日资格时，用合格恒生ETF（159920）承接海外权益目标，否则留现金。实际使用固定31只池内的这3只ETF。

2016年以来累计{pct(primary['cumulative_return'])}、年化{pct(primary['cagr'])}，同期人民币SPY全收益为{pct(bm_long['spy_total_cny']['cumulative_return'])}、{pct(bm_long['spy_total_cny']['cagr'])}；2021重新建仓年化{pct(recent['cagr'])}，同期基准{pct(bm_recent['spy_total_cny']['cagr'])}。这是已查看历史上的开发策略，投资准入仍未通过。

## 同口径比较

| 区间 | 对象 | 累计收益 | CAGR | 最大回撤 |
| --- | --- | ---: | ---: | ---: |
{comparison}

首次测试前定义“接近”：年化落后不超过2个百分点，累计收益至少为人民币SPY全收益的85%。策略资本100万元、扣除交易费用；SPY调整收盘包括分红再投资与基金内部费用，未扣个人税费。美元、人民币和不含分红的价格指数分别列示，没有用较低的价格收益替代全收益目标。

长区间从原始数据首日后的2016-01-05开始，两方起止日期一致，CAGR按实际年份计算。表中标普回撤使用全部美国交易日；曲线按中国报告日向后匹配，最多允许7个自然日报价空档。跨市场同日收盘时刻不同，美国收盘未用于中国当日信号。

基准原始响应、46次SPY分红、USD/CNY和来源时间戳已冻结；腾讯640日标普价格交叉误差小于0.000005%，现金分红递推与调整收盘累计收益差约0.087%。行情截至2026-09-11。

![净值与回撤](images/sp500_strategy_comparison.png)

## 可执行规则

1. 保留上市观察期、20日日均成交额至少5,000万元、当时已公告规模至少2亿元及退市/缺价检查；未回填最新规模。纳指资格不足时，才使用预先指定的恒生ETF代理，否则现金。
2. 每个中国交易日正式收盘后按65%海外权益、30%黄金目标生成次日开盘订单。偏离不到3个百分点可保持原权重，随后再检查风险约束。退出中的标的不重新买回；部分成交按真实现金和份额继续处理。
3. 观察波动率，不根据波动预测强制缩仓；目标总仓位最多95%，单只与同组下单上限85%，不做空、不加杠杆。价格变化可导致事后漂移，后续决策再调整。30%是本轮研究的历史回撤验收预算，原20%投资准入线继续单列；两者均非未来回撤保证。
4. 日双边换手预算30%，买入只使用卖出需求扣除后的剩余额度及现有现金，不预支未完成卖单。实际开盘再检查现金、集中度和容量；参与率最多为20日均成交额的2%，买入按100份整数倍，停牌/缺价或价格限制时拒绝成交。
5. 佣金单边0.03%、最低0.2元，基础滑点5bp，另加参与率冲击；现金不计息，另测最低5元佣金。QDII使用场内实际价格，包含汇率与溢价变化，没有拿境外指数替代成交价。

收益主要来自海外成长与黄金配置，不代表逐日跟踪标普500。原生产系统排除QDII、单只上限20%、目标波动15%，与本研究风险预算有明显差异；其投资门禁未修改。本方案也没有依赖尚未补齐资料的实时溢价过滤。

## 成本与执行压力

{table(result['stress'])}

6个情景是否均为正收益且回撤不超过30%：**{'是' if checks['all_execution_stress_positive_within_research_budget'] else '否'}**；是否都在原20%线内：**{'是' if checks['all_execution_stress_positive_within_20dd'] else '否'}**。表中完整显示收益接近能否维持，部分情景会明显落后基准。容量结论基于日线参与率及冲击模型，不等于开盘盘口实测。

整数份额、现金余量和调仓边界会让费用改变后续交易路径，因此各压力结果不是仅在同一净值曲线上扣一个固定费率，费用或资本与最终收益不必严格单调。

## 权重邻域与附加检查

海外权益55%/60%/65%/70%/75%，黄金补足95%总目标，保留5%现金。{passed}/{count}组同时满足收益接近及30%研究回撤要求，预定门槛为至少2/3通过。

{table(result['neighbors'])}

{table(result['additional'])}

重新开启波动择时与观察模式分别列示，不把不同风险预算下的变化全部归于选股能力。`split_ratio_5.0`将纳指2022-01-14的5.0019因子按新浪份额资料5.0重放，检验舍入差异；不能替代发行人的折算原始凭证。2016–2020为额外环境诊断，仍是已查看历史，不是新盲测。

## 年度与不利证据

| 年度 | 策略 | SPY全收益折算人民币 |
| --- | ---: | ---: |
{yearly}

滚动252交易日收益差最差{pct(diag['rolling_12m_worst_gap'])}，最好{pct(diag['rolling_12m_best_gap'])}；并非每年、每个窗口都能接近。月收益相关系数{diag['monthly_return_correlation_with_spy_cny']:.3f}。日收盘异步会影响日相关性，不能把低日相关直接解释为独立收益来源。

相对人民币SPY的日均超额收益21日块bootstrap（1,000次）95%区间为[{pct(ci['ci_low'])}, {pct(ci['ci_high'])}]。收益接近不等于显著战胜标普；固定现存31只池与34个含基线候选的选择均存在偏差。

## 工程验收与重放

- 冻结候选独立复跑净值最大差额{checks['reproduction_error_cny']:.8f}元。每个完整情景逐日检查现金与复权单位账本、信号早于成交、买入资格、整数份额和参与率。
- 86项测试通过，包括新增9项汇率/时间对齐、缺口拒绝、未来价格隔离、并列组、资格替代、分批建仓和风险模式检查。
- 原生产指针和代码指纹保持一致；独立保存研究结果，没有修改默认账户、自动下单或提交Git。最终核验记录在 `output/sp500_exploration/completion_verification.json`。

```powershell
# 校验冻结输入和规则后重放入选策略
.venv\\Scripts\\python.exe -m research.sp500.replay

# 重做完整验证
.venv\\Scripts\\python.exe -m research.sp500.validate --protocol research/sp500/validation_protocol_v3.json
```

规则：[selected_strategy.json](../research/sp500/selected_strategy.json)。完整结果：[results.json](../{directory.relative_to(BASE).as_posix()}/results.json)。计划：[中文实施记录](SP500_STRATEGY_EXPLORATION_PLAN.md)。

## 全部开发候选

均截至2026-09-11。不同起始区间明确列示，不跨区间直接排名；失败结果与前两轮未完成稳健性验收的记录均保留。

| 候选 | 起始日 | 累计收益 | CAGR | 最大回撤 | 接近同期人民币SPY |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(all_trials)}

**投资准入未通过。** 仍缺完整历史资料审计和126日真实前瞻；更高权益配置可能超过旧20%回撤线。历史目标达成不能替代新规则的前瞻账户和成交验证。
'''
    (BASE/"docs/SP500_STRATEGY_ACCEPTANCE.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": "docs/SP500_STRATEGY_ACCEPTANCE.md", "historical_goal_verified": completed,
                      "validation": str(directory), "image": str(image)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    make_report()
