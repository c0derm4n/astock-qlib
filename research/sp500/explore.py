"""独立研究：冻结候选，使用正式事件引擎回放，不发布生产指针。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import config
import universe
from src.market import Market, causal_features
from src.universe_filter import historical_universe
from src.portfolio import constrain_weights
from src.simulation import simulate, performance
from research.sp500.benchmark import aligned_wealth

BASE = Path(__file__).resolve().parents[2]


def frozen_market(source, include_qdii=True):
    source = Path(source)
    bars = pd.read_parquet(source / "market_snapshot.parquet")
    nav = pd.read_parquet(source / "nav_snapshot.parquet")
    meta = pd.read_parquet(source / "metadata_snapshot.parquet")
    with patch.object(config, "EXCLUDE_QDII", not include_qdii):
        eligible = historical_universe(bars, nav, meta)
    features, labels, ends = causal_features(bars)
    dates = pd.DatetimeIndex(bars.date.unique()).sort_values()
    audit = json.loads((source / "acceptance.json").read_text(encoding="utf-8"))["data_audit"]
    return Market(bars, eligible, features, labels, ends, dates, "frozen_source", audit, nav, meta)


def signal_matrix(market, candidate, model_scores):
    price = market.wide("adj_close")
    if candidate.get("signal") == "lightgbm":
        return model_scores.unstack("instrument").reindex(index=price.index, columns=price.columns)
    windows = candidate.get("lookbacks", [20, 60, 120])
    score = sum(price / price.shift(n) - 1 for n in windows) / len(windows)
    if candidate.get("signal") == "risk_momentum":
        volatility = price.pct_change(fill_method=None).rolling(60, min_periods=60).std() * np.sqrt(252)
        score = score / volatility.clip(lower=.05)
    return score.replace([np.inf, -np.inf], np.nan)


def bounded_targets(raw, current, returns, policy):
    """日风控和分批建仓；小额漂移沿用共享的二次风险核对。"""
    # 固定配置的观察模式仍保留单只/分组/总仓位约束，仅不按波动预测缩仓。
    risk_policy = ({**policy, "vol_target": float("inf")}
                   if policy.get("volatility_policy") == "observe_only" else policy)
    desired = constrain_weights(raw, returns, risk_policy)
    old = pd.Series(current, dtype=float)
    index = old.index.union(desired.index)
    old, new = old.reindex(index, fill_value=0), desired.reindex(index, fill_value=0)
    small = ((new - old).abs() < policy["rebalance_band"]) & old.gt(0) & new.gt(0)
    new.loc[small] = old.loc[small]
    new = constrain_weights(new, returns, risk_policy).reindex(index, fill_value=0)
    diff = new - old
    sells, buys = -diff.clip(upper=0).sum(), diff.clip(lower=0)
    budget = max(0., policy["max_daily_turnover"] - sells)
    if buys.sum() > budget:
        new = old + diff.clip(upper=0) + buys * budget / buys.sum()
    return new[new > 1e-9].to_dict()


def allocator(candidate, calendar, start):
    # 交易日历是已知制度信息；不读取未来价格来决定月末。
    dates = calendar[calendar >= pd.Timestamp(start)]
    periods = pd.Series(dates.to_period(candidate.get("rebalance_frequency", "M")))
    rebalance = set(dates[periods.ne(periods.shift(-1)).to_numpy()])
    rebalance.add(dates[0])
    execution_window = candidate.get("execution_window")
    window_dates = set()
    if execution_window is not None:
        if not isinstance(execution_window, int) or execution_window < 1:
            raise ValueError("调仓执行窗口必须为正整数")
        for day in rebalance:
            start_index = dates.get_loc(day)
            window_dates.update(dates[start_index:start_index+execution_window])
    raw = {}

    def select(scores, eligible, current, holding_days, returns, policy, *, mode="rank", exiting=()):
        nonlocal raw
        date = returns.index[-1]
        allowed = set(eligible) - set(exiting)
        if date in rebalance:
            raw = dict(candidate.get("weights", {}))
            if candidate["family"] in ("rotation", "hybrid"):
                pool = allowed - set(raw)
                if candidate["family"] == "hybrid":
                    pool = {s for s in pool if universe.get_asset_class(s[2:]) == "equity"}
                valid = scores.reindex(sorted(pool)).replace([np.inf, -np.inf], np.nan).dropna()
                if candidate.get("absolute_filter"):
                    valid = valid[valid > 0]
                valid = valid.sort_values(ascending=False, kind="stable")
                selected = []
                topk = candidate["topk"]
                for _, group in valid.groupby(valid, sort=False):
                    if len(selected) + len(group) > topk:
                        break
                    selected.extend(group.index)
                weight = candidate.get("satellite_weight", .95) / topk
                raw.update({s: weight for s in selected})
                # 绝对动量没有足额正信号时，剩余目标停泊国债；国债缺资格则现金。
                if candidate.get("absolute_filter"):
                    raw["SH511260"] = raw.get("SH511260", 0) + weight * (topk - len(selected))
        if execution_window is not None and date not in window_dates:
            # 调仓窗口之外仅做资格退出与风险缩减，允许合格持仓自然漂移。
            active = {s: w for s, w in current.items() if s in allowed}
            return bounded_targets(active, current, returns, policy)
        active = {s: w for s, w in raw.items() if s in allowed}
        # 资格不足时使用事先指定的池内代理；只按当日资格决定，不看未来收益。
        for original, alternatives in candidate.get("fallbacks", {}).items():
            if original not in allowed and original in raw:
                if sum(alternatives.values()) > 1 + 1e-10 or any(w < 0 for w in alternatives.values()):
                    raise ValueError("替代权重必须非负且总和不超过1")
                for symbol, fraction in alternatives.items():
                    if symbol in allowed:
                        active[symbol] = active.get(symbol, 0) + raw[original] * fraction
        return bounded_targets(active, current, returns, policy)
    return select


def run_candidate(market, scores, spec, policy, start, **stress):
    series = scores.stack(future_stack=True)
    series.index.names = ["datetime", "instrument"]
    series = series.loc[series.index.get_level_values(0) >= pd.Timestamp(start)]
    # 仅在此单进程研究作用域替换分配回调；订单/撮合/费用完全使用正式引擎。
    with patch("src.simulation.target_weights", allocator(spec, market.calendar, start)):
        return simulate(market, series, policy, **stress)


def run(spec_path=None, only=None):
    spec_path = Path(spec_path or Path(__file__).with_name("candidates.json"))
    registry = json.loads(spec_path.read_text(encoding="utf-8"))
    source_id = json.loads((BASE / "output/current_run.json").read_text(encoding="utf-8"))["run_id"]
    source = BASE / "output/runs" / source_id
    market = frozen_market(source)
    if registry["end"] != str(market.calendar[-1].date()):
        raise ValueError("登记截止日与冻结行情不一致")
    policy = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    model_scores = pd.read_pickle(source / "predictions_lightgbm.pkl")
    benchmark_dir = json.loads((BASE / "output/sp500_exploration/benchmarks/latest.json").read_text(encoding="utf-8"))["directory"]
    output = BASE / "output/sp500_exploration/runs" / pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "research_source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, output / "research_source" / path.name)
    (output / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"source_run": source_id, "benchmark_directory": benchmark_dir, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "registry_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "research_only": True, "trading_symbols": sorted(market.bars.symbol.unique()),
                "code": {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__), Path(__file__).with_name("benchmark.py"), BASE / "src/simulation.py", BASE / "src/portfolio.py"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    market.eligibility.to_parquet(output / "eligibility_31.parquet", index=False)
    summaries = {}
    for spec in registry["candidates"]:
        if only and spec["name"] not in only:
            continue
        began = time.monotonic()
        if spec["family"] == "existing":
            report = pd.read_csv(source / "report_lightgbm.csv", index_col=0, parse_dates=True)
            trades = pd.read_csv(source / "trades_lightgbm.csv")
            issues = {"source": "unchanged_production_replay"}
        else:
            scores = signal_matrix(market, spec, model_scores)
            candidate_policy = {**policy, **registry["execution"], **spec.get("policy_overrides", {})}
            report, trades, issues = run_candidate(market, scores, spec, candidate_policy, registry["start"])
        report.to_csv(output / f"report_{spec['name']}.csv")
        trades.to_csv(output / f"trades_{spec['name']}.csv", index=False)
        summary = performance(report)
        summary.update({"issues": issues, "trade_count": len(trades), "seconds": time.monotonic()-began,
                        "average_exposure": float((report.value / report.account).mean())})
        wealth = aligned_wealth(benchmark_dir, report.index)
        benchmarks = {}
        for key in wealth:
            returns = wealth[key].pct_change(fill_method=None).fillna(0)
            reference = pd.DataFrame({"return": returns, "cost": 0., "turnover": 0.})
            benchmarks[key] = performance(reference)
        benchmark = benchmarks["spy_total_cny"]
        summary["cagr_gap_vs_spy_cny"] = summary["cagr"] - benchmark["cagr"]
        summary["cumulative_ratio_vs_spy_cny"] = summary["cumulative_return"] / benchmark["cumulative_return"]
        comparison = registry["comparison"]
        summary["historical_target_met"] = (summary["cagr_gap_vs_spy_cny"] >= -comparison["max_cagr_shortfall"]
                                             and summary["cumulative_ratio_vs_spy_cny"] >= comparison["min_cumulative_ratio"])
        summaries[spec["name"]] = summary
        (output / "results.json").write_text(json.dumps({"candidates": summaries, "benchmarks": benchmarks}, ensure_ascii=False, indent=2), encoding="utf-8")
        wealth.to_csv(output / "benchmark_wealth.csv")
        print(f"{spec['name']}: 累计{summary['cumulative_return']:.2%} 年化{summary['cagr']:.2%} 回撤{summary['max_drawdown']:.2%} 达标={summary['historical_target_met']} 耗时{summary['seconds']:.1f}s", flush=True)
    (output.parent / "latest.json").write_text(json.dumps({"directory": str(output)}, indent=2), encoding="utf-8")
    print(f"探索完成：{output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    run(args.registry, args.only)
