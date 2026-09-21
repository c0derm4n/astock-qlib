"""冻结候选的压力/邻域/长历史与逐日账户审计。"""
from __future__ import annotations
import copy
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.sp500.explore import BASE, frozen_market, signal_matrix, run_candidate
from research.sp500.benchmark import aligned_wealth
from src.simulation import performance, block_bootstrap_mean


def ledger_audit(market, report, trades, policy, initial_capital):
    bars = market.bars.set_index(["date", "symbol"])
    eligible = market.eligibility.set_index(["date", "symbol"])
    adjclose = market.wide("adj_close").ffill(limit=5)
    units, cash = {}, initial_capital
    errors, cash_errors, timing, ineligible, odd_buys = [], [], [], [], []
    max_participation = 0.
    daily = {pd.Timestamp(date): group for date, group in trades.groupby("date")}
    for date, row in report.iterrows():
        for trade in daily.get(date, pd.DataFrame()).itertuples():
            signal = pd.Timestamp(trade.signal_date)
            if signal >= date:
                timing.append(str(date.date()))
            if trade.side == "buy" and not eligible.at[(signal, trade.symbol), "eligible"]:
                ineligible.append(trade.symbol)
            if trade.side == "buy" and abs(trade.quantity / 100 - round(trade.quantity / 100)) > 1e-7:
                odd_buys.append(trade.symbol)
            direction = 1 if trade.side == "buy" else -1
            factor = bars.at[(date, trade.symbol), "adj_factor"]
            units[trade.symbol] = units.get(trade.symbol, 0) + direction * trade.quantity / factor
            cash -= direction * trade.quantity * trade.price + trade.commission
            max_participation = max(max_participation, trade.participation)
        account = cash + sum(unit * adjclose.at[date, symbol] for symbol, unit in units.items() if abs(unit) > 1e-8)
        errors.append(abs(account-row.account))
        cash_errors.append(abs(cash-row.cash))
    result = {"rows": len(report), "trades": len(trades), "max_nav_error_cny": max(errors),
              "max_cash_error_cny": max(cash_errors), "timing_violations": timing,
              "ineligible_buys": ineligible, "nonlot_buys": odd_buys,
              "max_adv_participation": max_participation, "minimum_cash": float(report.cash.min())}
    if max(errors) > .01 or max(cash_errors) > .01 or timing or ineligible or odd_buys:
        raise ValueError(f"逐日账户审计失败：{result}")
    if max_participation > policy["max_adv_participation"] + 1e-8 or result["minimum_cash"] < -.01:
        raise ValueError("参与率/现金约束不一致")
    return result


def relative_metrics(report, benchmark_dir, comparison):
    metrics = performance(report)
    benchmark = aligned_wealth(benchmark_dir, report.index)
    years = (report.index[-1]-report.index[0]).days / 365.25
    bench_total = benchmark.spy_total_cny.iloc[-1] - 1
    bench_cagr = (1 + bench_total) ** (1/years) - 1
    metrics["benchmark_cumulative"] = float(bench_total)
    metrics["benchmark_cagr"] = float(bench_cagr)
    metrics["cagr_gap"] = float(metrics["cagr"] - bench_cagr)
    metrics["cumulative_ratio"] = float(metrics["cumulative_return"] / bench_total) if bench_total > 0 else None
    metrics["target_met"] = bool(metrics["cagr_gap"] >= -comparison["max_cagr_shortfall"] and
                                  bench_total > 0 and metrics["cumulative_ratio"] >= comparison["min_cumulative_ratio"])
    return metrics, benchmark


def validate(protocol_path=None):
    protocol_path = Path(protocol_path or Path(__file__).with_name("validation_protocol.json"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    exploration = BASE / "output/sp500_exploration/runs" / protocol["exploration_run"]
    registry = json.loads((exploration / "registry.json").read_text(encoding="utf-8"))
    manifest = json.loads((exploration / "manifest.json").read_text(encoding="utf-8"))
    source = BASE / "output/runs" / manifest["source_run"]
    market = frozen_market(source)
    primary = next(s for s in registry["candidates"] if s["name"] == protocol["primary_candidate"])
    policy = {**json.loads((source / "protocol.json").read_text(encoding="utf-8")), **registry["execution"],
              **primary.get("policy_overrides", {})}
    model_scores = pd.read_pickle(source / "predictions_lightgbm.pkl")
    scores = signal_matrix(market, primary, model_scores)
    benchmark_dir = manifest["benchmark_directory"]
    directory = BASE / "output/sp500_exploration/validation" / pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "research_source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, directory / "research_source" / path.name)
    pointer_before = (BASE / "output/current_run.json").read_bytes()
    spec = {"protocol": protocol, "candidate": primary, "execution_policy": policy,
            "source": str(source), "benchmark": benchmark_dir, "started_at": pd.Timestamp.now(tz="UTC").isoformat()}
    (directory / "frozen_validation.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {"base": {}, "stress": {}, "neighbors": {}, "additional": {}}

    def execute(name, section, candidate=primary, tested_market=market, tested_policy=policy, start=registry["start"], **kwargs):
        began = time.monotonic()
        frame, trades, issues = run_candidate(tested_market, scores, candidate, tested_policy, start, **kwargs)
        metrics, benchmark = relative_metrics(frame, benchmark_dir, registry["comparison"])
        metrics["execution_issues"] = issues
        metrics["trade_count"] = len(trades)
        metrics["average_exposure"] = float((frame.value/frame.account).mean())
        metrics["ledger"] = ledger_audit(tested_market, frame, trades, tested_policy,
                                          tested_policy["initial_capital"]*kwargs.get("capital_multiplier", 1))
        results[section][name] = metrics
        frame.to_csv(directory / f"report_{name}.csv")
        trades.to_csv(directory / f"trades_{name}.csv", index=False)
        (directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: 累计{metrics['cumulative_return']:.2%} 年化{metrics['cagr']:.2%} 回撤{metrics['max_drawdown']:.2%} 接近目标={metrics['target_met']} {time.monotonic()-began:.1f}s", flush=True)
        return frame, trades, benchmark

    base, base_trades, benchmark = execute("primary", "base")
    original = pd.read_csv(exploration / f"report_{primary['name']}.csv", index_col=0, parse_dates=True)
    reproduction_error = float((base.account-original.account).abs().max())
    if reproduction_error > .001:
        raise ValueError("冻结候选复现不一致")
    for scenario in protocol["stress"]:
        execute(scenario["name"], "stress", **scenario["kwargs"])
    for equity in protocol["weight_neighborhood"]["nasdaq"]:
        gold_values = ([.95-equity] if protocol["weight_neighborhood"].get("gold_mode") == "remaining"
                       else protocol["weight_neighborhood"]["gold"])
        for gold in gold_values:
            candidate = copy.deepcopy(primary)
            candidate["weights"] = {"SH513100": equity, "SH518880": gold, "SH511260": max(0., .95-equity-gold)}
            execute(f"nas{equity:.0%}_gold{gold:.0%}", "neighbors", candidate=candidate)
    policy_variants = protocol.get("policy_variants") or [("vol_target_15", {"vol_target": .15}), ("vol_target_25", {"vol_target": .25}),
                            ("band_0.02", {"rebalance_band": .02}), ("band_0.05", {"rebalance_band": .05}),
                            ("commission_min_5", {"minimum_commission": 5.})]
    for name, overrides in policy_variants:
        execute(name, "additional", tested_policy={**policy, **overrides})
    rounded = copy.deepcopy(market)
    rows = (rounded.bars.symbol == "SH513100") & rounded.bars.adj_factor.gt(1)
    rounded.bars.loc[rows, "adj_factor"] = 5.
    for field in ("open", "close"):
        rounded.bars.loc[rows, "adj_"+field] = rounded.bars.loc[rows, field] * 5.
    execute("split_ratio_5.0", "additional", tested_market=rounded)
    if pd.Timestamp(registry["start"]) <= pd.Timestamp("2016-01-05"):
        long_frame = base
        execute("restart_2021", "additional", start="2021-01-04")
    else:
        long_frame, _, _ = execute("sample_2016_onward", "additional", start="2016-01-04")
    early_frame = long_frame.loc[:"2020-12-31"]
    results["additional"]["sample_2016_2020"], _ = relative_metrics(early_frame, benchmark_dir, registry["comparison"])
    early_frame.to_csv(directory / "report_sample_2016_2020.csv")
    # 同一已研究时期的年度、滚动窗口、相关性与块置信区间，均保留不利结果。
    strategy_net = base["return"] - base.cost
    benchmark_net = benchmark.spy_total_cny.pct_change(fill_method=None).fillna(0)
    strategy_wealth = (1+strategy_net).cumprod()
    rolling = pd.DataFrame({"strategy_252d": strategy_wealth / strategy_wealth.shift(252) - 1,
                            "spy_cny_252d": benchmark.spy_total_cny / benchmark.spy_total_cny.shift(252) - 1})
    rolling["gap"] = rolling.strategy_252d - rolling.spy_cny_252d
    rolling.to_csv(directory / "rolling_12month.csv")
    source_manifest = json.loads((source / "run_meta.json").read_text(encoding="utf-8"))
    source_mismatches = [p for p, h in source_manifest["code"]["files"].items()
                         if hashlib.sha256((BASE/p).read_bytes()).hexdigest() != h]
    risk_budget = protocol.get("research_drawdown_budget", .2)
    checks = {"reproduction_error_cny": reproduction_error,
              "base_target_met": results["base"]["primary"]["target_met"],
              "all_execution_stress_positive_within_20dd": all(s["cumulative_return"] > 0 and s["max_drawdown"] >= -.2 for s in results["stress"].values()),
              "research_drawdown_budget": risk_budget,
              "all_execution_stress_positive_within_research_budget": all(s["cumulative_return"] > 0 and s["max_drawdown"] >= -risk_budget for s in results["stress"].values()),
              "neighbors_target_within_20dd": sum(s["target_met"] and s["max_drawdown"] >= -.2 for s in results["neighbors"].values()),
              "neighbors_target_within_research_budget": sum(s["target_met"] and s["max_drawdown"] >= -risk_budget for s in results["neighbors"].values()),
              "neighbor_count": len(results["neighbors"]),
              "production_pointer_unchanged": pointer_before == (BASE / "output/current_run.json").read_bytes(),
              "production_code_mismatches": source_mismatches}
    if not checks["production_pointer_unchanged"] or source_mismatches:
        raise ValueError("研究意外改变正式版本")
    monthly = pd.concat([strategy_wealth.resample("ME").last().pct_change(), benchmark.spy_total_cny.resample("ME").last().pct_change()], axis=1).dropna()
    results["diagnostics"] = {"return_correlation_with_spy_cny": float(strategy_net.corr(benchmark_net)),
                              "monthly_return_correlation_with_spy_cny": float(monthly.iloc[:,0].corr(monthly.iloc[:,1])),
                              "excess_daily_block_ci": block_bootstrap_mean(strategy_net-benchmark_net),
                              "rolling_12m_worst_gap": float(rolling.gap.min()),
                              "rolling_12m_best_gap": float(rolling.gap.max()),
                              "rolling_12m_positive_strategy_fraction": float(rolling.strategy_252d.dropna().gt(0).mean()),
                              "selection_trials": protocol.get("selection_trials", 20), "development_only": True}
    results["checks"] = checks
    (directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    (directory / "artifact_hashes.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    (directory.parent / "latest.json").write_text(json.dumps({"directory": str(directory)}, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(directory), "checks": checks}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    args = parser.parse_args()
    validate(args.protocol)
