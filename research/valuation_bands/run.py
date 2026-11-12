"""研究 CLI: check-data / register / run / report，以及合成数据与真实数据构建入口。"""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from . import calibration as calibration_mod
from . import data as data_mod
from . import engine as engine_mod

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "valuation_bands"
OUT_DIR = PROJECT_ROOT / "output" / "valuation_bands"
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
CODE_FILES = ("__init__.py", "data.py", "calibration.py", "signals.py", "engine.py", "run.py", "verify.py", "synthetic.py", "report.py", "fetch.py", "etfrun_data.py")

TERMINAL_STATUS = {
    "historical_run_complete",
    "synthetic_pipeline_validated",
    "historical_goals_passed",
    "data_or_scope_blocked",
    "failed",
}

CASE_FILE_COLUMNS: Dict[str, List[str]] = {
    "calibration.csv": ["strategy", "index_id", "metric", "year", "lookback_years", "q", "median_low_value",
                        "L1", "L2", "L3", "exit_level", "sample_count", "snapshot_hash", "params_hash", "created_at"],
    "signals.csv": ["strategy", "session", "timestamp", "phase_before", "phase_after_intent", "reason", "detail",
                    "value", "valuation_date", "valuation_available_at", "valuation_lag", "L1", "L2", "L3", "S",
                    "allowed_steps", "buy_steps_before", "buy_steps_after_intent", "orders", "skips", "checks",
                    "exec_session"],
    "orders.csv": ["order_id", "signal_date", "signal_timestamp", "execution_date", "side", "symbol", "reason",
                   "requested_quantity", "filled_quantity", "reference_price", "fill_price", "adv20_snapshot",
                   "commission", "impact", "status", "requires_symbol_flat", "note"],
    "trades.csv": ["trade_id", "order_id", "signal_date", "execution_date", "side", "symbol", "reason", "quantity",
                   "price", "notional", "commission", "impact", "slippage", "adv20", "cash_after",
                   "quantity_after", "status"],
    "actions.csv": ["event_id", "symbol", "type", "ex_date", "record_date", "payment_date", "cash_per_share",
                    "share_factor", "quantity_entitled", "amount", "recorded_session", "paid_session"],
    "daily_account.csv": ["session", "cash", "dividend_receivable", "quantity_SH510300", "quantity_SZ159915",
                          "quantity_SH512890", "price_SH510300", "price_SZ159915", "price_SH512890",
                          "equity_value", "nav", "equity_weight", "cash_share", "cost_basis",
                          "cumulative_fees", "cumulative_turnover", "stale_prices", "missing_data"],
    "holdings.csv": ["session", "SH510300_qty", "SH510300_value", "SZ159915_qty", "SZ159915_value",
                     "SH512890_qty", "SH512890_value", "receivable"],
    "annual_returns.csv": ["year", "return", "full_year", "start_nav", "end_nav", "sessions",
                           "start_session", "end_session"],
    "episodes.csv": ["symbol", "entry_session", "buy_notional", "sell_notional", "fees", "dividends", "status",
                     "pnl", "holding_days", "exit_session", "exit_notional", "exit_fees"],
}


# ---------------------------------------------------------------- 工具
def now_stamp() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%dT%H%M%S")


def run_dir_name() -> str:
    return f"{now_stamp()}-{secrets.token_hex(4)}"


def code_hashes(protocol_path=None) -> Dict[str, str]:
    hashes = {}
    for name in CODE_FILES:
        path = PACKAGE_DIR / name
        if path.exists():
            hashes[name] = data_mod.sha256_file(path)
    hashes["protocol.json"] = data_mod.sha256_file(Path(protocol_path) if protocol_path else PROTOCOL_PATH)
    return hashes


def protocol(path=None) -> Dict[str, object]:
    return json.loads((Path(path) if path else PROTOCOL_PATH).read_text(encoding="utf-8"))


def find_manifest(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (PROJECT_ROOT / path)
    snapshots = DATA_DIR / "snapshots"
    if not snapshots.exists():
        return None
    candidates = sorted([p for p in snapshots.iterdir() if (p / "manifest.json").exists()])
    return (candidates[-1] / "manifest.json") if candidates else None


def write_json(path: Path, payload: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame, columns: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False, encoding="utf-8")
        return
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    # 年度校准包含随年份变化的low_YYYY_*证据列，不能按固定表头丢弃。
    out = out[list(columns) + [c for c in out.columns if c not in columns]]
    out.to_csv(path, index=False, encoding="utf-8")


def required_series(protocol_doc: Mapping[str, object], case: Mapping[str, object]) -> List[str]:
    resolved = engine_mod.resolve_case(protocol_doc, case)
    series = [
        f"index_price:{resolved.index_id}",
        f"index_valuation:{resolved.index_id}:{resolved.metric}",
        f"etf_bars:{resolved.symbol}",
        f"etf_actions:{resolved.symbol}",
    ]
    if resolved.gate == "ratio":
        series.append("ratio_signals")
    if resolved.dividend_module:
        series.append("dividend_yield:H30269")
        series.append("etf_bars:SH512890")
        series.append("etf_actions:SH512890")
    return series


def available_series(data: data_mod.ResearchData) -> set:
    out = set()
    for frame_name, frame in data.frames.items():
        if frame is None or frame.empty:
            continue
        if frame_name == "index_price":
            for index_id in frame["index_id"].unique():
                out.add(f"index_price:{index_id}")
        elif frame_name == "index_valuation":
            for index_id, metric in frame[["index_id", "metric"]].drop_duplicates().itertuples(index=False):
                out.add(f"index_valuation:{index_id}:{metric}")
        elif frame_name == "etf_bars":
            for symbol in frame["symbol"].unique():
                out.add(f"etf_bars:{symbol}")
        elif frame_name == "dividend_yield":
            for index_id in frame["index_id"].unique():
                out.add(f"dividend_yield:{index_id}")
        else:
            out.add(frame_name)
    return out


def freeze_actual_end(data: data_mod.ResearchData, protocol_doc: Mapping[str, object], case=None) -> Optional[str]:
    """按合格输入冻结 actual_end; 必需估值序列缺失则无法冻结。"""
    ends: List[pd.Timestamp] = []
    strategies = ([protocol_doc["strategies"][case["strategy"]]] if case else
                  list(protocol_doc["strategies"].values()))
    for index_id, metric in [(s["index_id"], s["metric"]) for s in strategies]:
        frame = data.index_valuation(index_id, metric)
        if frame.empty:
            return None
        ends.append(pd.Timestamp(frame["trade_date"].max()))
    for index_id in [s["index_id"] for s in strategies]:
        frame = data.index_price(index_id)
        if frame.empty:
            return None
        ends.append(pd.Timestamp(frame["trade_date"].max()))
    for symbol in [s["etf"] for s in strategies]:
        frame = data.etf_bars(symbol)
        if frame.empty:
            return None
        ends.append(pd.Timestamp(frame["trade_date"].max()))
    requested = pd.Timestamp(protocol_doc["requested_end"])
    return str(min(min(ends), requested).date())


# ---------------------------------------------------------------- check-data
def cmd_check_data(args) -> int:
    manifest = find_manifest(args.manifest)
    protocol_doc = protocol(getattr(args, "protocol", None))
    out_dir = OUT_DIR / "checks" / f"{now_stamp()}-{secrets.token_hex(3)}"
    if manifest is None:
        report = {
            "status": "no_snapshot",
            "detail": "未找到 data/valuation_bands/snapshots/*/manifest.json",
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "protocol_id": protocol_doc["protocol_id"],
        }
        write_json(out_dir / "data_gap_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    loaded = data_mod.load_snapshot(manifest, verify_hashes=not args.skip_hash)
    gaps = data_mod.check_gaps(loaded, protocol_doc)
    gaps["manifest_path"] = str(manifest)
    gaps["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    gaps["protocol_id"] = protocol_doc["protocol_id"]
    write_json(out_dir / "data_gap_report.json", gaps)
    summary = {
        "run_dir": str(out_dir),
        "data_version": loaded.data_version,
        "dataset_kind": loaded.dataset_kind,
        "blocking_count": gaps["blocking_count"],
        "blocking_series": gaps["blocking_series"],
        "can_run_primary": gaps["can_run_primary"],
        "can_run_ratio_gate": gaps["can_run_ratio_gate"],
        "can_run_dividend": gaps["can_run_dividend"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------- register
def cmd_register(args) -> int:
    manifest = find_manifest(args.manifest)
    protocol_path = Path(getattr(args, "protocol", None) or PROTOCOL_PATH).resolve()
    protocol_doc = protocol(protocol_path)
    run_dir = OUT_DIR / "runs" / args.directory_name if args.directory_name else OUT_DIR / "runs" / run_dir_name()
    if run_dir.exists():
        print(f"运行目录已存在, 拒绝覆盖: {run_dir}")
        return 2
    run_dir.mkdir(parents=True, exist_ok=False)

    if manifest is None:
        registration = {
            "protocol_id": protocol_doc["protocol_id"],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "no_snapshot_registered",
            "investment_admission": False,
            "screenshot_reviewed": True,
            "cases": [
                {"id": case["id"], "group": case["group"], "status": "data_or_scope_blocked",
                 "detail": "no_snapshot"} for case in protocol_doc["cases"]
            ],
            "actual_end": None,
        }
        write_json(run_dir / "registration.json", registration)
        write_json(run_dir / "data_gap_report.json", {"status": "no_snapshot"})
        print(f"未找到数据快照, 已登记阻塞状态: {run_dir}")
        return 1

    loaded = data_mod.load_snapshot(manifest, verify_hashes=not args.skip_hash)
    gaps = data_mod.check_gaps(loaded, protocol_doc)
    actual_end = freeze_actual_end(loaded, protocol_doc)
    available = available_series(loaded)
    blocked_series = set(gaps.get("blocking_series") or [])
    cases = []
    for case in protocol_doc["cases"]:
        case_end = freeze_actual_end(loaded, protocol_doc, case)
        required = required_series(protocol_doc, case)
        missing = [s for s in required if s in blocked_series or
                   (s not in available and not s.startswith("etf_actions:"))]
        cases.append({
            **{k: case[k] for k in ("id", "group", "strategy", "kind")},
            "overrides": dict(case.get("overrides") or {}),
            "resolved": engine_mod.resolve_case(protocol_doc, case).as_dict(),
            "required_series": required,
            "actual_end": case_end,
            "missing_series": missing,
            "status": "registered" if not missing and case_end else "data_or_scope_blocked",
            "detail": "" if not missing and case_end else (
                f"missing:{','.join(missing)}" if missing else "actual_end_unfrozen"),
        })
    if actual_end is None:
        available_ends = [c["actual_end"] for c in cases if c["status"] == "registered"]
        actual_end = max(available_ends) if available_ends else None
    registration = {
        "protocol_id": protocol_doc["protocol_id"],
        "implementation_version": protocol_doc["implementation_version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": str(protocol_path),
        "protocol_hash": data_mod.sha256_file(protocol_path),
        "code_hashes": code_hashes(protocol_path),
        "data_version": loaded.data_version,
        "dataset_kind": loaded.dataset_kind,
        "manifest_path": str(manifest),
        "manifest_sha256": data_mod.sha256_file(manifest),
        "input_hashes": {entry["name"]: entry.get("sha256") for entry in loaded.manifest.get("files", [])},
        "data_asof": loaded.manifest.get("created_at"),
        "actual_end": actual_end,
        "requested_end": protocol_doc["requested_end"],
        "research_start": protocol_doc["research_start"],
        "initial_capital": protocol_doc["initial_capital"],
        "investment_admission": False,
        "screenshot_reviewed": True,
        "default_parameters_note": "全部执行、分批、仓位、风险参数为本方案补齐, 非截图原述",
        "data_evidence_grade": loaded.manifest.get("sources", []),
        "cases": cases,
        "benchmarks": protocol_doc["benchmarks"],
        "case_count": len(cases),
        "synthetic": bool(loaded.dataset_kind != "real"),
        "notes": protocol_doc.get("notes", []),
    }
    write_json(run_dir / "registration.json", registration)
    write_json(run_dir / "input_manifest.json", loaded.manifest)
    write_json(run_dir / "data_gap_report.json", gaps)
    print(json.dumps({
        "run_dir": str(run_dir),
        "registered_cases": sum(1 for c in cases if c["status"] == "registered"),
        "blocked_cases": sum(1 for c in cases if c["status"] == "data_or_scope_blocked"),
        "case_count": len(cases),
        "actual_end": actual_end,
        "dataset_kind": loaded.dataset_kind,
    }, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------- run
def _case_blocked(run_dir: Path, case: Mapping[str, object], detail: str, gaps: Mapping[str, object]):
    case_dir = run_dir / "cases" / str(case["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, columns in CASE_FILE_COLUMNS.items():
        write_frame(case_dir / name, pd.DataFrame(columns=columns), columns)
    status = {
        "case_id": case["id"],
        "group": case["group"],
        "strategy": case["strategy"],
        "status": "data_or_scope_blocked",
        "detail": detail,
        "missing_series": case.get("missing_series", []),
        "data_gap_report": "../../data_gap_report.json",
        "blocking_series": gaps.get("blocking_series", []),
        "metrics_available": False,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(case_dir / "case_status.json", status)
    write_json(case_dir / "metrics.json", {
        "case_id": case["id"], "status": "not_run_data_blocked", "detail": detail,
        "cagr": None, "max_drawdown": None, "worst_full_year": None,
        "goals": {"cagr_pass": None, "annual_loss_pass": None, "max_drawdown_pass": None, "joint_pass": None},
    })
    return status


def _write_case(run_dir: Path, result: engine_mod.CaseRunResult, case: Mapping[str, object]) -> Dict[str, object]:
    case_dir = run_dir / "cases" / str(case["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    write_frame(case_dir / "calibration.csv", result.calibration_rows, CASE_FILE_COLUMNS["calibration.csv"])
    write_frame(case_dir / "signals.csv", result.signals, CASE_FILE_COLUMNS["signals.csv"])
    write_frame(case_dir / "orders.csv", result.orders, CASE_FILE_COLUMNS["orders.csv"])
    write_frame(case_dir / "trades.csv", result.trades, CASE_FILE_COLUMNS["trades.csv"])
    write_frame(case_dir / "actions.csv", result.actions, CASE_FILE_COLUMNS["actions.csv"])
    write_frame(case_dir / "daily_account.csv", result.daily, CASE_FILE_COLUMNS["daily_account.csv"])
    write_frame(case_dir / "holdings.csv", result.holdings, CASE_FILE_COLUMNS["holdings.csv"])
    write_frame(case_dir / "annual_returns.csv", result.annual_returns, CASE_FILE_COLUMNS["annual_returns.csv"])
    write_frame(case_dir / "episodes.csv", result.episodes, CASE_FILE_COLUMNS["episodes.csv"])
    write_json(case_dir / "metrics.json", result.metrics)
    status = {
        "case_id": case["id"],
        "group": case["group"],
        "strategy": case["strategy"],
        "status": result.status,
        "detail": result.status_detail,
        "rows": {
            "signals": int(len(result.signals)),
            "orders": int(len(result.orders)),
            "trades": int(len(result.trades)),
            "daily": int(len(result.daily)),
        },
        "data_failures": result.data_failures[:20],
        "metrics_available": result.status in ("historical_run_complete", "historical_goals_passed", "synthetic_pipeline_validated"),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(case_dir / "case_status.json", status)
    return status


def cmd_run(args) -> int:
    run_dir = Path(args.directory)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    registration_path = run_dir / "registration.json"
    if not registration_path.exists():
        print(f"缺少登记文件: {registration_path}")
        return 2
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("synthetic") and not args.allow_synthetic:
        print("该 run 基于合成数据, 需显式 --allow-synthetic 才能执行(结果不作为绩效证据)")
        return 3
    protocol_path = Path(registration.get("protocol_path") or PROTOCOL_PATH)
    protocol_doc = protocol(protocol_path)
    if data_mod.sha256_file(protocol_path) != registration.get("protocol_hash"):
        print("协议文件哈希与登记不一致, 拒绝运行; 请新增 run 并说明变更原因")
        return 4
    current_code = code_hashes(protocol_path)
    if current_code != registration.get("code_hashes"):
        print("代码哈希与登记不一致, 拒绝运行; 请新增 run 并说明变更原因")
        return 5

    loaded = data_mod.load_snapshot(registration["manifest_path"])
    if data_mod.sha256_file(Path(registration["manifest_path"])) != registration.get("manifest_sha256"):
        print("输入 manifest 哈希与登记不一致, 拒绝运行")
        return 6
    gaps = data_mod.check_gaps(loaded, protocol_doc)

    groups = {args.group} if args.group != "all" else {"primary", "stress", "sensitivity", "auxiliary"}
    results: List[Dict[str, object]] = []
    for case in registration["cases"]:
        if case["group"] not in groups:
            continue
        case_dir = run_dir / "cases" / case["id"]
        status_path = case_dir / "case_status.json"
        if status_path.exists():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            if prior.get("status") in TERMINAL_STATUS:
                results.append(prior)
                continue
        if case["status"] == "data_or_scope_blocked" or registration.get("actual_end") is None:
            detail = case.get("detail") or "actual_end_unfrozen"
            results.append(_case_blocked(run_dir, case, detail, gaps))
            continue
        try:
            result = engine_mod.replay(
                loaded, protocol_doc, case, pd.Timestamp(case.get("actual_end") or registration["actual_end"]),
                start=pd.Timestamp(registration["research_start"]),
            )
        except Exception as error:  # 单 case 失败不掩盖其他 case
            case_dir.mkdir(parents=True, exist_ok=True)
            status = {
                "case_id": case["id"], "group": case["group"], "strategy": case["strategy"],
                "status": "failed", "detail": f"{type(error).__name__}: {error}",
                "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            write_json(case_dir / "case_status.json", status)
            results.append(status)
            continue
        results.append(_write_case(run_dir, result, case))

    # 基准
    bench_results = []
    if registration.get("actual_end"):
        for bench in protocol_doc["benchmarks"]:
            bench_dir = run_dir / "benchmarks" / bench["id"]
            if (bench_dir / "metrics.json").exists():
                bench_results.append(json.loads((bench_dir / "metrics.json").read_text(encoding="utf-8")))
                continue
            try:
                result = engine_mod.run_benchmark(
                    loaded, protocol_doc, bench, pd.Timestamp(registration["actual_end"]),
                    start=pd.Timestamp(registration["research_start"]), data_gaps=gaps,
                )
            except Exception as error:
                write_json(bench_dir / "case_status.json", {
                    "benchmark": bench["id"], "status": "failed", "detail": f"{type(error).__name__}: {error}"})
                continue
            bench_dir.mkdir(parents=True, exist_ok=True)
            write_frame(bench_dir / "daily_account.csv", result["daily"], CASE_FILE_COLUMNS["daily_account.csv"])
            write_frame(bench_dir / "trades.csv", result["trades"], CASE_FILE_COLUMNS["trades.csv"])
            write_frame(bench_dir / "actions.csv", result.get("actions", pd.DataFrame()),
                        CASE_FILE_COLUMNS["actions.csv"])
            write_json(bench_dir / "metrics.json", result["metrics"])
            bench_results.append(result["metrics"])

    results_path = run_dir / "results.json"
    existing = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    merged = {entry["case_id"]: entry for entry in existing.get("cases", [])}
    for entry in results:
        if entry.get("case_id"):
            merged[entry["case_id"]] = entry
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_kind": registration.get("dataset_kind"),
        "synthetic": registration.get("synthetic"),
        "actual_end": registration.get("actual_end"),
        "cases": [merged[key] for key in sorted(merged)],
        "benchmarks": bench_results,
    }
    write_json(results_path, payload)

    completion = {
        "updated_at": payload["updated_at"],
        "protocol_id": registration.get("protocol_id"),
        "dataset_kind": registration.get("dataset_kind"),
        "investment_admission": False,
        "total_cases": len(registration["cases"]),
        "by_status": _count_by_status(merged),
        "history_complete": all(
            merged.get(c["id"], {}).get("status") == "historical_run_complete"
            for c in registration["cases"] if c["status"] == "registered"
        ) and any(c["status"] == "registered" for c in registration["cases"]),
        "notes": [
            "implementation_passed 仅表示代码测试通过",
            "historical_run_complete 表示真实历史与对账完成",
            "data_or_scope_blocked 表示具体数据/范围缺口",
            "historical_goals_passed 仅表示当前口径历史三项联合达标",
        ],
    }
    write_json(run_dir / "completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


def _count_by_status(merged: Mapping[str, Mapping[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in merged.values():
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


# ---------------------------------------------------------------- report
def cmd_report(args) -> int:
    from .report import write_report

    run_dir = Path(args.directory)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    path = write_report(run_dir)
    print(f"报告已生成: {path}")
    return 0


# ---------------------------------------------------------------- 数据构建
def cmd_synthetic_data(args) -> int:
    from .synthetic import build_synthetic_dataset

    root = OUT_DIR / "synthetic" / "data" / (args.version or "synthetic-fixture-v1")
    manifest = build_synthetic_dataset(root, seed=args.seed)
    print(f"合成数据 manifest: {manifest}")
    return 0


def cmd_build_data(args) -> int:
    from .fetch import build_real_snapshot

    manifest = build_real_snapshot(DATA_DIR, requested_end=args.requested_end)
    print(f"真实数据 manifest: {manifest}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="research.valuation_bands.run", description="估值择时研究 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-data", help="只检查数据契约与缺口, 不修改正式库")
    p_check.add_argument("--manifest", default=None)
    p_check.add_argument("--protocol", default=None)
    p_check.add_argument("--skip-hash", action="store_true")
    p_check.set_defaults(func=cmd_check_data)

    p_reg = sub.add_parser("register", help="冻结参数/代码/输入哈希与全部case")
    p_reg.add_argument("--manifest", default=None)
    p_reg.add_argument("--protocol", default=None)
    p_reg.add_argument("--directory-name", default=None)
    p_reg.add_argument("--skip-hash", action="store_true")
    p_reg.set_defaults(func=cmd_register)

    p_run = sub.add_parser("run", help="运行已登记目录")
    p_run.add_argument("--directory", required=True)
    p_run.add_argument("--group", default="all", choices=["primary", "stress", "sensitivity", "auxiliary", "all"])
    p_run.add_argument("--allow-synthetic", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="生成中文报告与图表")
    p_rep.add_argument("--directory", required=True)
    p_rep.set_defaults(func=cmd_report)

    p_syn = sub.add_parser("synthetic-data", help="生成合成数据(仅流程验证)")
    p_syn.add_argument("--version", default=None)
    p_syn.add_argument("--seed", type=int, default=20261109)
    p_syn.set_defaults(func=cmd_synthetic_data)

    p_build = sub.add_parser("build-data", help="抓取公开免费数据并生成研究快照(不写正式库)")
    p_build.add_argument("--requested-end", default=None)
    p_build.set_defaults(func=cmd_build_data)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
