"""周频资产迁移：每周一次决策下的趋势门槛与波动预算，并核对决策日/成交日节奏。

只新增本实验目录，不修改既有冻结协议或产物。候选与政策逐项取自冻结基线归档，
节奏与下单封闭复用 research.goal30.weekly_goal 的既有实现。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.ex_nasdaq.core import market30, validate_candidate
from research.goal30 import risk_overlay
from research.goal30.audit import BASE, comparison
from research.goal30.cash_replay import audit_cash_ledger, simulate_cash
from research.goal30.weekly_goal import cadence_audit, weekly_overlay_allocator
from research.sp500.validate import relative_metrics
from src.artifacts import write_json_atomic

PROTOCOL = Path(__file__).with_name("weekly_transfer_protocol.json")
OUT = BASE / "output/goal30/weekly_transfer"
PARTIAL_YEARS = (2016, 2026)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def window_metrics(report, start, end):
    window = report.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(window) < 2:
        return None
    account = window["account"]
    years = (account.index[-1] - account.index[0]).days / 365.25
    annual = account.resample("YE").last()
    annual_returns = annual.pct_change(fill_method=None)
    annual_returns.iloc[0] = annual.iloc[0] / account.iloc[0] - 1
    complete = {int(k.year): float(v) for k, v in annual_returns.items() if int(k.year) not in PARTIAL_YEARS}
    worst = min(complete.items(), key=lambda item: item[1]) if complete else None
    return {
        "start": str(account.index[0].date()), "end": str(account.index[-1].date()),
        "observations": int(len(account)),
        "cumulative_return": float(account.iloc[-1] / account.iloc[0] - 1),
        "cagr": float((account.iloc[-1] / account.iloc[0]) ** (1 / years) - 1) if years > 0 else None,
        "max_drawdown": float((account / account.cummax() - 1).min()),
        "average_exposure": float((window["value"] / window["account"]).mean()),
        "sum_daily_turnover": float(window["turnover"].sum()),
        "worst_complete_year": ({"year": worst[0], "return": worst[1]} if worst else None),
        "annual_returns": {str(int(k.year)): float(v) for k, v in annual_returns.items()},
    }


def goal_checks(metrics, goals):
    conservative, advanced = goals["conservative"], goals["advanced"]
    worst = metrics["worst_complete_year"]
    drawdown_loss = -float(metrics["max_drawdown"])
    require(0 <= drawdown_loss < 1, "最大回撤必须以负收益表示")
    annual_ok = worst is not None and worst["return"] >= conservative["worst_complete_year"]
    cagr_advanced_ok = metrics["cagr"] >= advanced["cagr"]
    drawdown_ok = drawdown_loss <= advanced["max_drawdown"]
    return {
        "conservative_met": bool(metrics["cagr"] >= conservative["cagr"]
                                 and annual_ok),
        "advanced_met": bool(cagr_advanced_ok and drawdown_ok),
        "joint_met": bool(cagr_advanced_ok and annual_ok and drawdown_ok),
        "conservative_target": conservative, "advanced_target": advanced,
    }


def register():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    baseline = BASE / spec["baseline_archive"]
    frozen = json.loads((baseline / "frozen_protocol.json").read_text(encoding="utf-8"))
    require(spec["candidates"][0] == frozen["candidate"], "首个候选必须与冻结基线候选逐项一致")
    require(spec["start"] == frozen["protocol"]["start"] and spec["end"] == frozen["protocol"]["end"],
            "必须沿用冻结基线的同一区间")
    require(spec["cadence"] == "weekly_first_session", "周度节奏口径缺失")
    for candidate in spec["candidates"]:
        validate_candidate(candidate)
    require(len(spec["candidates"]) == len({c["name"] for c in spec["candidates"]}) == 2, "候选缺失或重复")
    require(len(spec["variants"]) == len({v["name"] for v in spec["variants"]}) == 4, "方案缺失或重复")
    require(len(spec["scenarios"]) == len({s["name"] for s in spec["scenarios"]}) == 3, "压力情景缺失或重复")
    for variant in spec["variants"]:
        risk_overlay.risk_policy(frozen["policy"], variant)
    data_hashes = {path: expected for path, expected in frozen["input_hashes"].items()
                   if path.startswith("output/")}
    require(len(data_hashes) == 21, "冻结数据输入条目数与预期不符")
    for path, expected in data_hashes.items():
        require(digest(BASE / path) == expected, f"冻结数据输入已变化：{path}")
    stale = {}
    for path, expected in frozen["input_hashes"].items():
        full = BASE / path
        if not full.exists() or digest(full) != expected:
            stale[path] = {"expected": expected, "actual": digest(full) if full.exists() else None}
    directory = OUT / (pd.Timestamp.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    registration = {
        "registered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protocol": spec, "policy": frozen["policy"], "actions": frozen["actions"],
        "source": spec["source"], "benchmark": spec["benchmark"], "comparison": spec["comparison"],
        "data_input_hashes": data_hashes, "stale_source_inputs": stale,
        "stale_source_note": "基线归档的 src/research 哈希早于 2026-09-21 内核清理；数据输入已单独硬校验。",
        "daily_counterpart": "output/goal30/asset_transfer（同候选同参数的日频结果，用于量化周频代价）",
        "cases": len(spec["candidates"]) * len(spec["variants"]) * len(spec["scenarios"]),
    }
    write_json_atomic(directory / "registration.json", registration)
    write_json_atomic(OUT / "latest_registration.json", {
        "directory": str(directory), "registration_sha256": digest(directory / "registration.json")})
    print(json.dumps({"registered": str(directory), "cases": registration["cases"]}, ensure_ascii=False), flush=True)
    return directory


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT) and directory.is_dir(), "只能运行已登记周频迁移目录")
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    for path, expected in registration["data_input_hashes"].items():
        require(digest(BASE / path) == expected, f"登记后冻结数据输入发生变化：{path}")
    with (directory / "started.json").open("x", encoding="utf-8") as output:
        json.dump({"started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()}, output)
    spec = registration["protocol"]
    market = market30(BASE / registration["source"])
    prices = market.wide("adj_close")
    cases = {}
    for candidate in spec["candidates"]:
        for variant in spec["variants"]:
            for scenario in spec["scenarios"]:
                name = "__".join((candidate["name"], variant["name"], scenario["name"]))
                began = time.monotonic()
                policy = risk_overlay.risk_policy(registration["policy"], variant)
                trace = []
                selector = weekly_overlay_allocator(prices, variant, trace)
                with patch("research.goal30.cash_replay.allocator", selector), \
                     patch("research.goal30.cash_replay.plan_orders", selector.plan):
                    report, trades, actions, holdings, issues = simulate_cash(
                        market, candidate, policy, spec["start"], registration["actions"],
                        **{key: value for key, value in scenario.items() if key != "name"})
                _, benchmark = relative_metrics(report, BASE / registration["benchmark"], registration["comparison"])
                metrics = comparison(report.account, benchmark.spy_total_cny)
                ledger = audit_cash_ledger(market, report, trades, actions, policy["initial_capital"])
                cadence = cadence_audit(market.calendar, spec["start"], trace, trades)
                full = window_metrics(report, spec["start"], spec["end"])
                checks = goal_checks(full, spec["goals"])
                cases[name] = {
                    **metrics, "candidate": candidate["name"], "variant": variant, "scenario": scenario,
                    "ledger": ledger, "cadence": cadence, "issues": issues, "goals": checks,
                    "full_window": full,
                    "slices": {item["name"]: window_metrics(report, item["start"], item["end"])
                               for item in spec["slices"]},
                    "seconds": time.monotonic() - began,
                }
                for prefix, frame in (("report", report), ("trades", trades), ("holdings", holdings),
                                      ("signals", pd.DataFrame(trace)), ("benchmark", benchmark)):
                    frame.to_csv(directory / f"{prefix}_{name}.csv", index=prefix in ("report", "benchmark"))
                write_json_atomic(directory / "results.json", {"cases": cases, "investment_admission": False})
                print(json.dumps({"case": name, "cagr": full["cagr"], "max_drawdown": full["max_drawdown"],
                                  "worst_year": full["worst_complete_year"], "decision_days": cadence["decision_days"],
                                  "conservative": checks["conservative_met"], "advanced": checks["advanced_met"]},
                                 ensure_ascii=True), flush=True)
    expected = registration["cases"]
    require(len(cases) == expected, f"登记情景未全部完成：{len(cases)}/{expected}")
    write_json_atomic(directory / "completion.json", {
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "cases": len(cases),
        "results_sha256": digest(directory / "results.json"),
        "registration_sha256": digest(directory / "registration.json")})
    print(json.dumps({"completed": str(directory), "cases": len(cases)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", type=Path)
    args = parser.parse_args()
    register() if args.register else run(args.run)
