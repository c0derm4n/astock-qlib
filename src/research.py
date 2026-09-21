"""固定协议研究、历史压力验收和独立生产训练；不根据测试收益改协议。"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import config
from backend.artifacts import write_json_atomic
from src.market import load_market
from src.models import walk_forward_models, fit_production
from src.runs import new_run, publish, read_protocol, code_fingerprint
from src.simulation import simulate, performance, block_bootstrap_mean
from src.utils import load_names, picks_table


def ic_analysis(scores, labels, protocol):
    df = pd.concat([scores.rename("score"), labels.rename("label")], axis=1).dropna()
    values = []
    for date, group in df.groupby(level="datetime"):
        if len(group) >= 3 and group["score"].std() > 1e-12 and group["label"].std() > 1e-12:
            values.append({"date": date, "ic": group["score"].corr(group["label"]),
                           "rank_ic": group["score"].corr(group["label"], method="spearman")})
    series = pd.DataFrame(values)
    if series.empty:
        return {"IC均值": None, "ICIR": None, "RankIC均值": None, "RankICIR": None}, {}
    metrics = {}
    for column, mean_name, ir_name in (("ic", "IC均值", "ICIR"), ("rank_ic", "RankIC均值", "RankICIR")):
        metrics[mean_name] = float(series[column].mean())
        std = series[column].std()
        metrics[ir_name] = float(series[column].mean()/std) if std > 0 else None
    ci = block_bootstrap_mean(series["rank_ic"], protocol["bootstrap_block_days"], protocol["bootstrap_samples"], protocol["seed"])
    return metrics, ci


def evaluate_gate(metrics, stress, excess_ci, protocol, forward_days):
    complete = {year: value for year, value in metrics["annual_returns"].items()
                if int(year) < pd.Timestamp(metrics["end"]).year}
    tests = {
        "historical_net_positive": metrics["cumulative_return"] > 0,
        "drawdown_within_budget": metrics["max_drawdown"] >= -protocol["max_drawdown_acceptance"],
        "majority_full_years_positive": bool(complete) and sum(v > 0 for v in complete.values()) > len(complete)/2,
        "double_slippage_positive": stress["slippage_2"]["cumulative_return"] > 0,
        "execution_stress_survives": bool(stress) and all(
            m["cumulative_return"] > 0 and m["max_drawdown"] >= -protocol["max_drawdown_acceptance"]
            for m in stress.values()),
        "excess_vs_momentum_confident": excess_ci["ci_low"] is not None and excess_ci["ci_low"] > 0,
        "forward_observation_sufficient": forward_days >= protocol["min_forward_days"],
        "historical_universe_audited": bool(protocol["historical_universe_audited"]),
        "corporate_actions_audited": bool(protocol["corporate_actions_audited"]),
    }
    return {"approved": all(tests.values()), "checks": tests,
            "failed_checks": [key for key, passed in tests.items() if not passed],
            "forward_days": forward_days, "mode": "paper_only" if not all(tests.values()) else "eligible_for_manual_review"}


def stress_scenarios(protocol, full_stress=True):
    scenarios = [("slippage_2", {"slippage_multiplier": 2.})]
    if full_stress:
        for value in protocol["stress_slippage_multipliers"]:
            if value not in (1., 2.):
                scenarios.append((f"slippage_{value:g}", {"slippage_multiplier": value}))
        for value in protocol["stress_capital_multipliers"]:
            if value != 1.:
                scenarios.append((f"capital_{value:g}", {"capital_multiplier": value}))
        for value in protocol["stress_fill_fractions"]:
            if value != 1.:
                name = "partial_fills" if value == .5 else f"fill_{value:g}"
                scenarios.append((name, {"fill_fraction": value}))
        for value in protocol["stress_execution_delays"]:
            if value != 1:
                name = "delay_one_day" if value == 2 else f"delay_{value-1}_days"
                scenarios.append((name, {"delay": value}))
    return scenarios


def index_benchmark(dates):
    """真沪深300指数买入持有对照：与策略同窗口，不计费用与换手。

    指数日线是不含分红的价格指数，与 ETF 后复权全收益口径不同，故仅作对照展示，
    不参与投资准入门禁。指数缺失或未覆盖整个窗口时返回 None，由调用方跳过该行。
    """
    code = getattr(config, "BENCHMARK_INDEX_TS", "")
    if not code or len(dates) < 2:
        return None
    from src.datasource import DataStore
    with DataStore() as store:
        frame = store.load_index(code)
    if frame.empty:
        return None
    closes = pd.Series(frame["close"].to_numpy(dtype=float),
                       index=pd.to_datetime(frame["date"])).sort_index()
    closes = closes[~closes.index.duplicated()]
    if closes.empty or closes.index[0] > dates[0] or closes.index[-1] < dates[-1]:
        return None
    closes = closes.reindex(dates).ffill()
    if closes.isna().any():
        return None
    net = closes.pct_change(fill_method=None).fillna(0.)
    return pd.DataFrame({"return": net, "cost": 0., "turnover": 0.}, index=dates)


def run_research(*, as_of=None, full_stress=True):
    protocol = read_protocol()
    code = code_fingerprint()
    # 每份协议首次运行记录规则冻结时间；当前已有数据不重复标记为全新样本外。
    protocol_hash = __import__("hashlib").sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    freeze_file = config.OUTPUT_DIR / "protocol_freezes" / f"{protocol_hash}-{code['fingerprint']}.json"
    if not freeze_file.exists():
        write_json_atomic(freeze_file, {"protocol_hash": protocol_hash,
                          "frozen_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()})
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    market = load_market(as_of=as_of, horizon=protocol["label_horizon"])
    run_id, directory = new_run(protocol)
    print(f"研究运行 {run_id}；数据 {market.audit['start']}~{market.audit['end']}；14项因果特征", flush=True)
    market.bars.to_parquet(directory / "market_snapshot.parquet", index=False)
    market.nav.to_parquet(directory / "nav_snapshot.parquet", index=False)
    market.meta.to_parquet(directory / "metadata_snapshot.parquet", index=False)
    import shutil
    import importlib.metadata
    for relative in code["files"]:
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.BASE_DIR / relative, destination)
    write_json_atomic(directory / "dependencies.json", {name: importlib.metadata.version(name) for name in
        ("pandas", "numpy", "lightgbm", "scikit-learn", "duckdb", "pyarrow", "exchange-calendars")})
    market.eligibility.to_parquet(directory / "historical_universe.parquet", index=False)
    forecasts, folds = walk_forward_models(market, protocol)
    write_json_atomic(directory / "folds.json", {"folds": folds})
    reports, summaries = {}, {}
    for name, scores in forecasts.items():
        report, trades, issues = simulate(market, scores, protocol)
        reports[name] = report
        summaries[name] = performance(report, protocol["annualization_days"])
        summaries[name]["execution_limits"] = issues
        report.to_csv(directory / f"report_{name}.csv")
        trades.to_csv(directory / f"trades_{name}.csv", index=False)
        scores.to_pickle(directory / f"predictions_{name}.pkl")
        print(f"{name}: 扣费累计 {summaries[name]['cumulative_return']:.2%} 最大回撤 {summaries[name]['max_drawdown']:.2%}", flush=True)
    primary = protocol["primary_model"]
    for name, mode in (("equal", "equal"), ("balanced", "balanced"), ("csi300_etf", "benchmark")):
        baseline_policy = dict(protocol)
        if mode == "benchmark":
            baseline_policy.update(max_weight=1., max_group_weight=1., max_equity_weight=1., vol_target=10.)
        report, trades, issues = simulate(market, forecasts[primary], baseline_policy, mode=mode)
        reports[name], summaries[name] = report, performance(report, protocol["annualization_days"])
        report.to_csv(directory / f"report_{name}.csv")
        print(f"对照 {name}: 扣费累计 {summaries[name]['cumulative_return']:.2%}", flush=True)
    index_report = index_benchmark(reports[primary].index)
    if index_report is None:
        print(f"对照 csi300_index: 跳过（{config.BENCHMARK_INDEX_TS} 指数日线缺失或未覆盖窗口）", flush=True)
    else:
        reports["csi300_index"] = index_report
        summaries["csi300_index"] = performance(index_report, protocol["annualization_days"])
        index_report.to_csv(directory / "report_csi300_index.csv")
        print(f"对照 csi300_index: 累计 {summaries['csi300_index']['cumulative_return']:.2%}", flush=True)
    stress = {}
    scenarios = stress_scenarios(protocol, full_stress)
    for name, kwargs in scenarios:
        report, trades, issues = simulate(market, forecasts[primary], protocol, **kwargs)
        report.to_csv(directory / f"stress_{name}.csv")
        trades.to_csv(directory / f"stress_trades_{name}.csv", index=False)
        stress[name] = performance(report, protocol["annualization_days"])
        stress[name]["execution_limits"] = issues
        print(f"压力 {name}: {stress[name]['cumulative_return']:.2%}", flush=True)
    neighbors = {}
    if full_stress:
        for topk in protocol["parameter_topk"]:
            for hold in protocol["parameter_hold_days"]:
                if topk == protocol["topk"] and hold == protocol["hold_days"]:
                    neighbors[f"top{topk}_hold{hold}"] = summaries[primary]
                    continue
                report, _, _ = simulate(market, forecasts[primary], dict(protocol, topk=topk, hold_days=hold))
                neighbors[f"top{topk}_hold{hold}"] = performance(report, protocol["annualization_days"])
                print(f"邻域 Top{topk}/hold{hold}: {neighbors[f'top{topk}_hold{hold}']['cumulative_return']:.2%}", flush=True)
    ablations = {}
    if full_stress and protocol.get("portfolio_style") == "core_satellite":
        for name, changed in (("without_core", dict(protocol, portfolio_style="rank")),
                              ("without_rank_buffer", dict(protocol, rank_buffer=0))):
            ablation, _, _ = simulate(market, forecasts[primary], changed)
            ablations[name] = performance(ablation, protocol["annualization_days"])
            ablation.to_csv(directory / f"ablation_{name}.csv")
            print(f"消融 {name}: {ablations[name]['cumulative_return']:.2%}", flush=True)
    report = reports[primary]
    net = report["return"] - report["cost"]
    excess = net - (reports["momentum"]["return"] - reports["momentum"]["cost"])
    excess_ci = block_bootstrap_mean(excess, protocol["bootstrap_block_days"], protocol["bootstrap_samples"], protocol["seed"])
    ic, ic_ci = ic_analysis(forecasts[primary], market.labels, protocol)
    from src.forward import forward_evidence
    observed = forward_evidence(protocol_hash, code["fingerprint"])
    forward_days = observed["days"]
    gate = evaluate_gate(summaries[primary], stress, excess_ci, protocol, forward_days)
    gate["checks"]["forward_performance_acceptable"] = (observed["return"] is not None and observed["return"] > 0
        and observed["max_drawdown"] >= -protocol["max_drawdown_acceptance"] and observed["execution_aligned"])
    gate["checks"]["full_stress_executed"] = bool(full_stress)
    gate["checks"]["parameter_neighborhood_survives"] = bool(neighbors) and sum(
        m["cumulative_return"] > 0 and m["max_drawdown"] >= -protocol["max_drawdown_acceptance"]
        for m in neighbors.values()) / len(neighbors) >= 2/3
    gate["checks"]["market_data_fresh"] = (pd.Timestamp.now().normalize() - market.calendar[-1]).days <= config.MAX_DATA_STALE_DAYS
    gate["checks"]["market_coverage_complete"] = not market.audit["stale_symbols"] and not market.audit["lifecycle_symbols_without_bars"]
    if not full_stress:
        gate["approved"] = False
        gate["failed_checks"].append("full_stress_executed")
    gate["failed_checks"] = [key for key, passed in gate["checks"].items() if not passed]
    gate["approved"] = all(gate["checks"].values())
    gate["mode"] = "eligible_for_manual_review" if gate["approved"] else "paper_only"
    bundle = fit_production(market, protocol)
    bundle.update({"run_id": run_id, "protocol_hash": protocol_hash, "code_fingerprint": code["fingerprint"]})
    with (directory / "production_model.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    report = report.copy()
    report["bench"] = reports["csi300_etf"]["return"] - reports["csi300_etf"]["cost"]
    report.to_csv(directory / "backtest_report.csv")
    forecasts[primary].to_pickle(directory / "predictions.pkl")
    latest = forecasts[primary].index.get_level_values("datetime").max()
    picks_table(forecasts[primary].xs(latest, level="datetime"), load_names(), protocol["topk"]).to_csv(directory / "latest_picks.csv", index=False, encoding="utf-8-sig")
    acceptance = {"run_id": run_id, "gate": gate, "summaries": summaries, "stress": stress,
                  "parameter_neighbors": neighbors, "excess_vs_momentum_block_ci": excess_ci,
                  "rank_ic_block_ci": ic_ci, "data_audit": market.audit, "protocol_freeze": freeze,
                  "forward_observation": observed, "ablations": ablations}
    write_json_atomic(directory / "acceptance.json", acceptance)
    write_json_atomic(directory / "train_metrics.json", {"ic": ic, "backtest": {
        "start": summaries[primary]["start"], "end": summaries[primary]["end"], "topk": protocol["topk"],
        "strategy_cum_return_pct": summaries[primary]["cumulative_return"]*100,
        "strategy_annual_pct": summaries[primary]["cagr"]*100,
        "strategy_max_drawdown_pct": summaries[primary]["max_drawdown"]*100,
        "hs300_cum_return_pct": summaries["csi300_etf"]["cumulative_return"]*100,
        "hs300_index_cum_return_pct": (summaries["csi300_index"]["cumulative_return"]*100
                                       if "csi300_index" in summaries else None),
        "eqw_cum_return_pct": summaries["equal"]["cumulative_return"]*100},
        "finished_at": pd.Timestamp.now().isoformat(), "data_version": run_id, "investment_gate": gate})
    manifest = {"schema_version": 2, "run_id": run_id, "data_version": market.fingerprint,
                "trained_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "code": code,
                "protocol_hash": protocol_hash, "protocol": protocol, "data_audit": market.audit,
                "calendar_start": market.audit["start"], "calendar_end": market.audit["end"],
                "ic_metrics": ic, "investment_gate": gate, "production": {k: bundle[k] for k in ("trained_as_of", "train_signal_end", "train_label_end", "train_rows", "best_iteration")},
                "tradable": market.eligibility.loc[(market.eligibility["date"] == market.calendar[-1]) & market.eligibility["eligible"], "symbol"].tolist()}
    manifest["n_tradable"] = len(manifest["tradable"])
    publish(run_id, directory, manifest)
    print(f"研究产物已发布：{directory}；投资准入：{'通过' if gate['approved'] else '未通过，仅模拟研究'}", flush=True)
    return acceptance, directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", help="仅使用此日期之前的行情")
    parser.add_argument("--quick", action="store_true", help="开发冒烟：跳过完整压力测试，不可通过投资验收")
    args = parser.parse_args()
    run_research(as_of=args.as_of, full_stress=not args.quick)


if __name__ == "__main__":
    main()
