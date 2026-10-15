"""刷新原候选的最长窗口与周频指标，所有产物隔离于生产。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from src.market import load_market
from src.simulation import performance, simulate
from research.strategy_ab_audit import calendar_metrics

BASE = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("refresh_protocol.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(report, policy):
    result = performance(report, policy["annualization_days"])
    result["max_drawdown_loss"] = abs(result["max_drawdown"])
    annual = result["annual_returns"]
    complete = {y: v for y, v in annual.items() if 2018 <= int(y) <= 2025}
    year = min(complete, key=complete.get)
    result["worst_complete_year"] = {"year": int(year), "return": complete[year]}
    result["partial_years"] = ["2026"]
    result["common_2018_2025"] = calendar_metrics(report)
    result["current_joint_numeric_goal_met"] = (result["cagr"] >= .135 and
        result["max_drawdown_loss"] <= .4 and complete[year] >= -.05)
    result["ten_year_validated"] = False
    result["investment_admission"] = False
    return result


def verify_weekly_trades(trades, calendar, first_sessions, delay):
    if trades.empty:
        return {"trades": 0, "weekly_signal_only": True, "execution_delay_sessions": delay}
    signals = pd.to_datetime(trades.signal_date)
    executions = pd.to_datetime(trades.date)
    if not signals.isin(first_sessions).all():
        raise ValueError("发现周首交易日以外的信号")
    expected = calendar[calendar.searchsorted(signals) + delay]
    if not np.array_equal(expected.to_numpy(), executions.to_numpy()):
        raise ValueError("成交时间与登记延迟不一致")
    if not (trades.participation <= .020000001).all():
        raise ValueError("成交参与率超过2%")
    return {"trades": len(trades), "weekly_signal_only": True,
            "execution_delay_sessions": delay, "maximum_participation": float(trades.participation.max())}


def run(source):
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = Path(source)
    if not source.is_absolute():
        source = BASE / source
    if not (source / "completion.json").exists():
        raise ValueError("模型训练归档未完成")
    policy = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if policy["candidate_models"] != spec["model_candidates"]:
        raise ValueError("候选模型与登记不一致")
    pointer_hash = digest(BASE / "output/current_run.json")
    run_id = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = BASE / "output/two_line_research/line1" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    market = load_market(horizon=policy["label_horizon"])
    if str(market.calendar[-1].date()) != spec["window_end"]:
        raise ValueError("登记截止日与数据不一致")
    files = [PROTOCOL, Path(__file__), source / "protocol.json", source / "folds.json",
             *[source / f"predictions_{name}.pkl" for name in spec["model_candidates"]],
             BASE / "src/models.py", BASE / "src/market.py", BASE / "src/simulation.py", BASE / "src/portfolio.py"]
    hashes = {str(p.relative_to(BASE)).replace("\\", "/"): digest(p) for p in files}
    registration = {"registered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protocol": spec, "policy": policy, "model_source": str(source.relative_to(BASE)),
        "input_hashes": hashes, "market_fingerprint": market.fingerprint,
        "production_pointer_sha256": pointer_hash, "research_only": True}
    write_json_atomic(directory / "registration.json", registration)
    print(json.dumps({"registered": str(directory), "market_end": spec["window_end"]}), flush=True)
    dates = market.calendar[market.calendar >= pd.Timestamp(spec["window_start"])]
    weeks = dates.to_period("W-SUN")
    first_sessions = dates[np.r_[True, weeks[1:] != weeks[:-1]]]
    cases = {}
    for name in spec["model_candidates"] + spec["reference_accounts"]:
        forecasts = pd.read_pickle(source / f"predictions_{name if name != 'balanced' else 'lightgbm'}.pkl")
        if name != "balanced":
            report = pd.read_csv(source / f"report_{name}.csv", index_col="date", parse_dates=True)
            cases[name + "__daily_base"] = {"kind": "candidate", "metrics": metrics(report, policy),
                "source_report": str((source / f"report_{name}.csv").relative_to(BASE))}
        weekly = forecasts.loc[forecasts.index.get_level_values("datetime").isin(first_sessions)]
        for scenario in spec["scenarios"]:
            case = name + "__weekly_" + scenario["name"]
            report, trades, issues = simulate(market, weekly, policy,
                mode="balanced" if name == "balanced" else "rank",
                slippage_multiplier=scenario["slippage_multiplier"], delay=scenario["delay"])
            check = verify_weekly_trades(trades, market.calendar, first_sessions, scenario["delay"])
            if (report.cash < -1e-6).any():
                raise ValueError("现金为负")
            report.to_csv(directory / f"report_{case}.csv")
            trades.to_csv(directory / f"trades_{case}.csv", index=False)
            cases[case] = {"kind": "reference" if name == "balanced" else "candidate",
                "metrics": metrics(report, policy), "issues": issues, "execution_verification": check,
                "weekly_signal_dates": int(weekly.index.get_level_values("datetime").nunique())}
            write_json_atomic(directory / "results.json", {"research_only": True,
                "cases": cases, "production_pointer_unchanged": digest(BASE / "output/current_run.json") == pointer_hash})
            print(json.dumps({"case": case, "cagr": cases[case]["metrics"]["cagr"],
                "worst_year": cases[case]["metrics"]["worst_complete_year"],
                "max_drawdown_loss": cases[case]["metrics"]["max_drawdown_loss"]}), flush=True)
    if any(digest(BASE / path) != expected for path, expected in hashes.items()):
        raise ValueError("登记输入在运行中改变")
    if digest(BASE / "output/current_run.json") != pointer_hash:
        raise ValueError("生产指针发生变化")
    write_json_atomic(directory / "completion.json", {"completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "cases": len(cases), "registration_sha256": digest(directory / "registration.json"),
        "results_sha256": digest(directory / "results.json"), "production_pointer_unchanged": True})
    write_json_atomic(BASE / "output/two_line_research/line1/latest.json", {"directory": str(directory.relative_to(BASE))})
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    run(arguments.source)
