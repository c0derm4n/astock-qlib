"""中文报告与图表输出。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd


def _load(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _status_label(status: str) -> str:
    return {
        "historical_run_complete": "真实历史完成",
        "synthetic_pipeline_validated": "合成流程验证",
        "data_or_scope_blocked": "数据/范围阻塞",
        "failed": "失败",
        "registered": "已登记未运行",
        "implementation_passed": "实现测试通过",
        "historical_goals_passed": "本口径历史达标",
    }.get(status, status or "未知")


def _plot(run_dir: Path, cases: List[Mapping[str, object]]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                                  "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return None
    curves = []
    for case in cases:
        if case.get("metrics_available") is False:
            continue
        path = run_dir / "cases" / str(case["case_id"]) / "daily_account.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "nav" not in frame:
            continue
        curves.append((str(case["case_id"]), pd.to_datetime(frame["session"]), frame["nav"].astype(float)))
    if not curves:
        return None
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8.0), dpi=140, sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for label, sessions, nav in curves:
        normalized = nav / nav.iloc[0]
        ax.plot(sessions, normalized, label=label, linewidth=1.0)
        drawdown = normalized / normalized.cummax() - 1.0
        ax2.plot(sessions, drawdown, linewidth=0.8, alpha=0.8)
    registration = _load(run_dir / "registration.json")
    title = ("合成数据流程验证（不得作为绩效）" if registration.get("synthetic") else "估值择时历史研究")
    ax.set_title(title + ": 归一化净值与回撤(各 case 独立100万元账户)")
    ax.set_ylabel("净值(起点=1)")
    ax.grid(alpha=0.3)
    ax2.set_ylabel("相对峰值回撤")
    ax2.grid(alpha=0.3)
    if len(curves) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = run_dir / "equity_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def write_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    registration = _load(run_dir / "registration.json")
    results = _load(run_dir / "results.json")
    gaps = _load(run_dir / "data_gap_report.json")
    verification = _load(run_dir / "verification.json")
    completion = _load(run_dir / "completion.json")

    cases = results.get("cases", [])
    synthetic = bool(registration.get("synthetic"))
    lines: List[str] = []
    lines.append("# 估值择时研究报告(自动生成)")
    lines.append("")
    lines.append(f"- 运行目录: `{run_dir.name}`")
    lines.append(f"- 协议: `{registration.get('protocol_id')}` v{registration.get('implementation_version')}")
    lines.append(f"- 数据版本: `{registration.get('data_version')}` (kind={registration.get('dataset_kind')})")
    lines.append(f"- 区间: {registration.get('research_start')} → {registration.get('actual_end') or '未冻结'}"
                 f"(请求终点 {registration.get('requested_end')})")
    lines.append(f"- 初始资金: {registration.get('initial_capital')} 元 / 每 case 独立账户")
    lines.append("- 投资准入: **false**(历史研究不等同实盘准入)")
    if synthetic:
        lines.append("")
        lines.append("> **本 run 基于合成数据(synthetic_fixture), 仅用于流程与账本验证, 不得作为任何绩效或策略结论。**")

    if gaps:
        lines.append("")
        lines.append("## 1. 数据缺口")
        lines.append("")
        lines.append(f"- 阻塞项数量: {gaps.get('blocking_count', '—')}")
        blocked_series = gaps.get("blocking_series") or []
        if blocked_series:
            lines.append(f"- 阻塞序列: {', '.join(f'`{s}`' for s in blocked_series)}")
        for gap in gaps.get("gaps", [])[:20]:
            flag = "阻塞" if gap.get("blocking") else "非阻塞"
            lines.append(f"- [{flag}] `{gap.get('series')}` · {gap.get('code')} · {gap.get('detail')}")
        lines.append("")
        lines.append("完整缺口证据见 `data_gap_report.json`。")

    lines.append("")
    lines.append("## 2. 场景状态")
    lines.append("")
    lines.append("| case | 分组 | 状态 | 明细 |")
    lines.append("| --- | --- | --- | --- |")
    for case in cases:
        lines.append(f"| {case.get('case_id')} | {case.get('group')} | {_status_label(case.get('status'))} "
                     f"| {str(case.get('detail') or '')[:70]} |")
    if completion:
        lines.append("")
        lines.append("状态统计: " + ", ".join(f"{k}={v}" for k, v in (completion.get("by_status") or {}).items()))

    metrics_cases = []
    for case in cases:
        if case.get("metrics_available") is False:
            continue
        path = run_dir / "cases" / str(case["case_id"]) / "metrics.json"
        payload = _load(path)
        if payload.get("cagr") is not None:
            metrics_cases.append(payload)
    if metrics_cases:
        lines.append("")
        lines.append("## 3. 账户指标(含现金、费用、分红应收)")
        lines.append("")
        lines.append("| case | 期末资金 | 累计收益 | CAGR | 最差完整年 | 最大回撤 | 年均权益 | 成交数 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for metrics in metrics_cases:
            lines.append(
                f"| {metrics.get('case_id')} | {_fmt(metrics.get('final_nav'), 2)} | "
                f"{_pct(metrics.get('cumulative_return'))} | {_pct(metrics.get('cagr'))} | "
                f"{_pct(metrics.get('worst_full_year'))} | {_pct(metrics.get('max_drawdown'))} | "
                f"{_pct(metrics.get('avg_equity_weight'))} | {metrics.get('trade_count')} |"
            )
        lines.append("")
        lines.append("三项联合目标判定(cagr≥13.5%, 最差完整年≥-5%, 最大回撤≤40%):")
        lines.append("")
        lines.append("| case | CAGR | 年度亏损 | 回撤 | 联合 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for metrics in metrics_cases:
            goals = metrics.get("goals", {})
            lines.append(f"| {metrics.get('case_id')} | {'通过' if goals.get('cagr_pass') else '未通过'} | "
                         f"{'通过' if goals.get('annual_loss_pass') else '未通过'} | "
                         f"{'通过' if goals.get('max_drawdown_pass') else '未通过'} | "
                         f"{'通过' if goals.get('joint_pass') else '未通过'} |")
        lines.append("")
        lines.append("> 风险两项优先于CAGR; 未过线即未过线, 不用高CAGR补偿。")

    benchmarks = results.get("benchmarks") or []
    if benchmarks:
        lines.append("")
        lines.append("## 4. 基准")
        lines.append("")
        lines.append("| 基准 | 期末资金 | CAGR | 最差完整年 | 最大回撤 | 实际初始仓位 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for bench in benchmarks:
            info = bench.get("benchmark", {})
            lines.append(f"| {info.get('id', bench.get('case_id'))} | {_fmt(bench.get('final_nav'), 2)} | "
                         f"{_pct(bench.get('cagr'))} | {_pct(bench.get('worst_full_year'))} | "
                         f"{_pct(bench.get('max_drawdown'))} | {_pct(info.get('actual_initial_position'))} |")

    if verification:
        lines.append("")
        lines.append("## 5. 独立对账")
        lines.append("")
        lines.append(f"- 抽查 case 数: {verification.get('checked_cases')}, 通过 {verification.get('passed')}, "
                     f"失败 {verification.get('failed')}, 跳过 {verification.get('skipped')}")
        lines.append(f"- 容差: 金额 {verification.get('cases', [{}])[0].get('tolerances', {}).get('amount', 0.01) if verification.get('cases') else 0.01} 元, "
                     f"份额 1e-8")
        for case in verification.get("cases", []):
            if case.get("checked"):
                lines.append(f"  - {case['case']}: {case['status']}, 最大净值差异 {_fmt(case.get('max_nav_diff'), 6)} 元, "
                             f"最大份额差异 {_fmt(case.get('max_quantity_diff'), 10)}")

    lines.append("")
    lines.append("## 6. 口径与限制")
    lines.append("")
    lines.append("- 信号仅在每周首个中国交易日19:00评估, 最早下一交易日开盘执行; 收盘估值不按同日收盘成交。")
    lines.append("- 对未知发布时间使用 `assumed_next_session_19` 延迟模型，并检查显式可得时间；这不能证明历史供应商序列不存在财报重述或成分回溯。`historical_reconstructed`结果仍属探索性证据。")
    lines.append("- 年度校准只用信号年之前的5个完整自然年, 年内在首次评估时冻结。")
    lines.append("- 现金利息与ETF印花税按0; 佣金双边0.0003且单笔最低0.20元; 滑点、冲击与ADV参与率见协议。")
    lines.append("- 截图自述的持有期与收益不是本项目已核实绩效, 本报告不宣称复现作者收益。")
    lines.append("")
    lines.append("## 7. 产物索引")
    lines.append("")
    for name in ("registration.json", "input_manifest.json", "data_gap_report.json", "results.json",
                 "verification.json", "completion.json", "report.md"):
        lines.append(f"- `{name}`")
    lines.append("- `cases/<case_id>/`: calibration/signals/orders/trades/actions/daily_account/holdings/"
                 "annual_returns/episodes/metrics/case_status")
    chart = _plot(run_dir, cases)
    if chart:
        lines.append(f"- `{chart}`")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
