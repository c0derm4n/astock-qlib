"""独立验收: 从落盘成交、公司行动与原价重新计算每日账户与指标。

本模块不导入 signals/engine 的决策函数, 只读取产物与原始行情, 用于交叉核对。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .data import load_snapshot, sha256_file

AMOUNT_TOL = 0.01      # 元
SHARE_TOL = 1e-8       # 份
RATIO_TOL = 1e-6       # 指标相对容差
PRICE_TOL = 1e-8       # 原价绝对容差


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _recompute_case(
    case_dir: Path,
    bars: Mapping[str, pd.DataFrame],
    initial: float,
    action_sources: Optional[Mapping[str, pd.DataFrame]] = None,
    max_price_carry_sessions: int = 5,
) -> Dict[str, object]:
    daily = _read(case_dir / "daily_account.csv")
    trades = _read(case_dir / "trades.csv")
    actions = _read(case_dir / "actions.csv")
    orders = _read(case_dir / "orders.csv")
    if daily.empty or "session" not in daily:
        return {"status": "no_daily", "case_dir": case_dir.name, "checked": False}
    daily["session"] = pd.to_datetime(daily["session"])
    symbols = [c.replace("quantity_", "") for c in daily.columns if c.startswith("quantity_")]

    if not actions.empty:
        actions = actions.copy()
        actions["event_id"] = actions["event_id"].astype(str)
        for column in ("ex_date", "record_date", "payment_date", "recorded_session", "paid_session"):
            if column in actions:
                actions[column] = pd.to_datetime(actions[column], errors="coerce")
        action_by_event = {
            str(row["event_id"]): row for _, row in actions.iterrows()
        }
    else:
        action_by_event = {}

    if action_sources is None:
        source_actions = actions.copy()
    else:
        frames = []
        # CSV保留了统一的三资产数量列，零列不代表账户实际参与该资产。
        # 从真实交易和行动日志确定账户范围，仍完整核验已交易资产的源事件。
        active_symbols = set(trades["symbol"]) if "symbol" in trades else set()
        if "symbol" in actions:
            active_symbols.update(actions["symbol"])
        for symbol in active_symbols:
            frame = action_sources.get(symbol, pd.DataFrame()).copy()
            if frame.empty:
                continue
            frame["symbol"] = symbol
            for column in ("ex_date", "record_date", "payment_date"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
            if "event_id" in frame:
                frame["event_id"] = frame["event_id"].astype(str)
            frames.append(frame)
        source_actions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not source_actions.empty:
        source_actions = source_actions[
            source_actions["symbol"].isin(symbols)
            & (source_actions["ex_date"] >= daily["session"].iloc[0])
            & (source_actions["ex_date"] <= daily["session"].iloc[-1])
        ].copy()

    cash = initial
    qty: Dict[str, int] = {s: 0 for s in symbols}
    receivable = 0.0
    rows: List[Dict[str, object]] = []
    quantity_history: Dict[pd.Timestamp, Dict[str, int]] = {}
    pending_dividends: List[Dict[str, object]] = []
    expected_paid_session: Dict[str, str] = {}
    action_amount_diff = 0.0
    action_entitlement_diff = 0.0
    action_source_ids = set()

    trade_by_day: Dict[pd.Timestamp, List[pd.Series]] = {}
    if not trades.empty:
        trades = trades.copy()
        trades["execution_date"] = pd.to_datetime(trades["execution_date"])
        for _, row in trades.iterrows():
            trade_by_day.setdefault(row["execution_date"], []).append(row)
    action_by_ex: Dict[pd.Timestamp, List[pd.Series]] = {}
    if not source_actions.empty:
        for row in source_actions.itertuples(index=False):
            action_source_ids.add(str(row.event_id))
            action_by_ex.setdefault(pd.Timestamp(row.ex_date), []).append(row)

    max_notional_diff = 0.0
    max_trade_cash_after_diff = 0.0
    max_trade_qty_after_diff = 0.0
    t1_violations = 0
    negative_quantity_events = 0
    sessions = list(daily["session"])
    last_close: Dict[str, tuple[Optional[pd.Timestamp], float]] = {}
    unpriceable_holding_days = 0

    for session in sessions:
        for action in action_by_ex.get(session, []):
            symbol = str(action.symbol)
            event_id = str(action.event_id)
            action_type = str(action.type)
            recorded = action_by_event.get(event_id)
            if action_type == "split":
                factor = float(action.share_factor)
                if np.isfinite(factor) and factor > 0:
                    qty[symbol] = int(round(qty.get(symbol, 0) * factor))
            elif action_type == "dividend":
                record_date = pd.Timestamp(action.record_date) if pd.notna(action.record_date) else None
                if record_date is None:
                    position = sessions.index(session)
                    record_date = sessions[max(0, position - 1)]
                entitled = int(quantity_history.get(pd.Timestamp(record_date), {}).get(symbol, 0))
                cash_per_share = float(action.cash_per_share) if pd.notna(action.cash_per_share) else 0.0
                expected_amount = entitled * cash_per_share
                recorded_amount = (
                    float(recorded["amount"])
                    if recorded is not None and pd.notna(recorded.get("amount"))
                    else float("nan")
                )
                recorded_entitled = (
                    int(recorded["quantity_entitled"])
                    if recorded is not None and pd.notna(recorded.get("quantity_entitled"))
                    else -1
                )
                action_amount_diff = max(
                    action_amount_diff,
                    abs(expected_amount - recorded_amount)
                    if np.isfinite(recorded_amount) else float("inf"),
                )
                action_entitlement_diff = max(action_entitlement_diff, abs(entitled - recorded_entitled))
                payment_date = pd.Timestamp(action.payment_date) if pd.notna(action.payment_date) else None
                if expected_amount > 0:
                    receivable += expected_amount
                    pending_dividends.append({
                        "event_id": event_id,
                        "amount": expected_amount,
                        "payment_date": payment_date,
                    })
                if payment_date is None or expected_amount <= 0:
                    expected_paid_session[event_id] = ""
                else:
                    payment_session = next((s for s in sessions if s >= payment_date), None)
                    expected_paid_session[event_id] = (
                        str(pd.Timestamp(payment_session).date()) if payment_session is not None else ""
                    )

        due = [
            item for item in pending_dividends
            if item["payment_date"] is not None and item["payment_date"] <= session
        ]
        for item in due:
            receivable -= float(item["amount"])
            cash += float(item["amount"])
        pending_dividends = [item for item in pending_dividends if item not in due]

        bought_today: Dict[str, int] = {symbol: 0 for symbol in symbols}
        for trade in trade_by_day.get(session, []):
            symbol = str(trade["symbol"])
            trade_qty = int(trade["quantity"])
            trade_price = float(trade["price"])
            notional = float(trade["notional"])
            fee = float(trade["commission"])
            max_notional_diff = max(max_notional_diff, abs(trade_qty * trade_price - notional))
            if trade["side"] == "buy":
                cash -= notional + fee
                qty[symbol] = qty.get(symbol, 0) + trade_qty
                bought_today[symbol] = bought_today.get(symbol, 0) + trade_qty
            else:
                available = max(0, qty.get(symbol, 0) - bought_today.get(symbol, 0))
                if trade_qty > available:
                    t1_violations += 1
                cash += notional - fee
                qty[symbol] = qty.get(symbol, 0) - trade_qty
            if qty.get(symbol, 0) < 0:
                negative_quantity_events += 1
            if "cash_after" in trade and pd.notna(trade["cash_after"]):
                max_trade_cash_after_diff = max(
                    max_trade_cash_after_diff,
                    abs(cash - float(trade["cash_after"])),
                )
            if "quantity_after" in trade and pd.notna(trade["quantity_after"]):
                max_trade_qty_after_diff = max(
                    max_trade_qty_after_diff,
                    abs(qty.get(symbol, 0) - int(trade["quantity_after"])),
                )
        equity = 0.0
        prices: Dict[str, float] = {}
        for symbol in symbols:
            frame = bars.get(symbol)
            price = None
            if frame is not None and not frame.empty:
                hit = frame[frame["trade_date"] == session]
                if len(hit):
                    price = float(hit.iloc[0]["close"])
            if price is not None and np.isfinite(price) and price > 0:
                last_close[symbol] = (session, price)
            elif symbol in last_close:
                price_date, prior_price = last_close[symbol]
                lag = sessions.index(session) - sessions.index(pd.Timestamp(price_date))
                price = prior_price if lag <= max_price_carry_sessions else None
            else:
                price = None
            if price is None:
                price = 0.0
                if qty.get(symbol, 0) > 0:
                    unpriceable_holding_days += 1
            prices[symbol] = price
            equity += qty.get(symbol, 0) * price
        nav = cash + equity + receivable
        rows.append({
            "session": session,
            "cash": cash,
            "receivable": receivable,
            "equity_value": equity,
            "nav": nav,
            "equity_weight": equity / nav if nav > 0 else 0.0,
            "cash_share": cash / nav if nav > 0 else 0.0,
            **{f"qty_{s}": qty.get(s, 0) for s in symbols},
            **{f"price_{s}": prices[s] for s in symbols},
        })
        quantity_history[session] = dict(qty)

    recomputed = pd.DataFrame(rows)

    cash_diff = float((recomputed["cash"] - daily["cash"].astype(float)).abs().max())
    nav_diff = float((recomputed["nav"] - daily["nav"].astype(float)).abs().max())
    qty_diff = 0
    for symbol in symbols:
        qty_diff = max(qty_diff, float((recomputed[f"qty_{symbol}"] - daily[f"quantity_{symbol}"].astype(float)).abs().max()))
    rec_diff = float((recomputed["receivable"] - daily["dividend_receivable"].astype(float)).abs().max())
    equity_diff = 0.0
    weight_diff = 0.0
    cash_share_diff = 0.0
    if "equity_value" in daily:
        equity_diff = float((recomputed["equity_value"] - daily["equity_value"].astype(float)).abs().max())
    if "equity_weight" in daily:
        weight_diff = max(
            weight_diff,
            float((recomputed["equity_weight"] - daily["equity_weight"].astype(float)).abs().max()),
        )
    if "cash_share" in daily:
        cash_share_diff = float((recomputed["cash_share"] - daily["cash_share"].astype(float)).abs().max())
    price_diff = 0.0
    for symbol in symbols:
        column = f"price_{symbol}"
        if column in daily:
            price_diff = max(
                price_diff,
                float((recomputed[column] - daily[column].astype(float)).abs().max()),
            )

    paid_date_mismatches = 0
    for event_id, expected_value in expected_paid_session.items():
        recorded = action_by_event.get(event_id)
        recorded_value = ""
        if recorded is not None and pd.notna(recorded.get("paid_session")):
            recorded_value = str(pd.Timestamp(recorded["paid_session"]).date())
        if recorded_value != expected_value:
            paid_date_mismatches += 1
    recorded_action_ids = set(action_by_event)
    action_event_id_mismatches = len(action_source_ids.symmetric_difference(recorded_action_ids))

    nav = recomputed["nav"].to_numpy(dtype=float)
    metrics_path = case_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    days = max(1, int((recomputed["session"].iloc[-1] - recomputed["session"].iloc[0]).days))
    cagr = (nav[-1] / initial) ** (365.2425 / days) - 1
    series = np.concatenate([[initial], nav])
    dd = series / np.maximum.accumulate(series) - 1
    max_dd = float(-dd.min())
    cagr_diff = abs(cagr - float(metrics.get("cagr") or 0.0))
    dd_diff = abs(max_dd - float(metrics.get("max_drawdown") or 0.0))
    final_nav_diff = abs(float(recomputed["nav"].iloc[-1]) - float(metrics.get("final_nav") or 0.0))

    annual_rows = []
    previous_end = initial
    frame = recomputed.copy()
    frame["year"] = frame["session"].dt.year
    for year in sorted(frame["year"].unique()):
        block = frame[frame["year"] == year]
        end_nav = float(block["nav"].iloc[-1])
        annual_rows.append({
            "return": end_nav / previous_end - 1 if previous_end > 0 else float("nan"),
            "full_year": bool(
                block["session"].iloc[0].month == 1
                and block["session"].iloc[0].day <= 5
                and block["session"].iloc[-1].month == 12
            ),
        })
        previous_end = end_nav
    full_returns = [row["return"] for row in annual_rows if row["full_year"]]
    worst_full_year = min(full_returns) if full_returns else None
    metrics_worst = metrics.get("worst_full_year")
    worst_full_year_diff = (
        0.0 if worst_full_year is None and metrics_worst is None
        else abs(float(worst_full_year) - float(metrics_worst))
        if worst_full_year is not None and metrics_worst is not None
        else float("inf")
    )
    metrics_max_weight = metrics.get("max_equity_weight")
    max_weight_diff = (
        0.0 if metrics_max_weight is None
        else abs(float(recomputed["equity_weight"].max()) - float(metrics_max_weight))
    )

    passed = bool(
        cash_diff <= AMOUNT_TOL and nav_diff <= AMOUNT_TOL and rec_diff <= AMOUNT_TOL
        and qty_diff <= SHARE_TOL and equity_diff <= AMOUNT_TOL
        and weight_diff <= RATIO_TOL and cash_share_diff <= RATIO_TOL
        and price_diff <= PRICE_TOL
        and max_notional_diff <= AMOUNT_TOL and max_trade_cash_after_diff <= AMOUNT_TOL
        and max_trade_qty_after_diff <= SHARE_TOL
        and action_amount_diff <= AMOUNT_TOL and action_entitlement_diff == 0
        and paid_date_mismatches == 0 and action_event_id_mismatches == 0
        and unpriceable_holding_days == 0 and t1_violations == 0
        and negative_quantity_events == 0
        and cagr_diff <= RATIO_TOL and dd_diff <= RATIO_TOL
        and final_nav_diff <= AMOUNT_TOL and worst_full_year_diff <= RATIO_TOL
        and max_weight_diff <= RATIO_TOL
    )
    return {
        "case": case_dir.name,
        "checked": True,
        "status": "passed" if passed else "failed",
        "max_cash_diff": cash_diff,
        "max_nav_diff": nav_diff,
        "max_quantity_diff": qty_diff,
        "max_receivable_diff": rec_diff,
        "max_equity_diff": equity_diff,
        "max_price_diff": price_diff,
        "max_weight_diff": weight_diff,
        "max_cash_share_diff": cash_share_diff,
        "max_action_amount_diff": action_amount_diff,
        "max_action_entitlement_diff": action_entitlement_diff,
        "action_event_id_mismatches": action_event_id_mismatches,
        "paid_date_mismatches": paid_date_mismatches,
        "unpriceable_holding_days": unpriceable_holding_days,
        "t1_violations": t1_violations,
        "negative_quantity_events": negative_quantity_events,
        "max_trade_notional_diff": max_notional_diff,
        "max_trade_cash_after_diff": max_trade_cash_after_diff,
        "max_trade_quantity_after_diff": max_trade_qty_after_diff,
        "cagr_recomputed": cagr,
        "max_drawdown_recomputed": max_dd,
        "cagr_diff": cagr_diff,
        "max_drawdown_diff": dd_diff,
        "final_nav_diff": final_nav_diff,
        "worst_full_year_diff": worst_full_year_diff,
        "max_weight_metric_diff": max_weight_diff,
        "rows": int(len(daily)),
        "trades": int(len(trades)),
        "orders": int(len(orders)),
        "tolerances": {"amount": AMOUNT_TOL, "shares": SHARE_TOL, "ratio": RATIO_TOL},
    }


def verify_run(run_dir, write: bool = True) -> Dict[str, object]:
    run_dir = Path(run_dir)
    registration = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
    initial = float(registration["initial_capital"])
    snapshot = load_snapshot(registration["manifest_path"])
    protocol_path = Path(registration.get("protocol_path") or Path(__file__).resolve().parent / "protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    max_price_carry_sessions = int(protocol["execution"]["max_price_carry_sessions"])
    bars: Dict[str, pd.DataFrame] = {}
    action_sources: Dict[str, pd.DataFrame] = {}
    for symbol in ("SH510300", "SZ159915", "SH512890"):
        frame = snapshot.etf_bars(symbol)
        if not frame.empty:
            frame = frame.copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            bars[symbol] = frame
        actions = snapshot.etf_actions(symbol)
        if not actions.empty:
            actions = actions.copy()
            for column in ("ex_date", "record_date", "payment_date"):
                actions[column] = pd.to_datetime(actions[column], errors="coerce")
        action_sources[symbol] = actions

    cases: List[Dict[str, object]] = []
    for case in registration["cases"]:
        case_dir = run_dir / "cases" / case["id"]
        status_path = case_dir / "case_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") not in ("historical_run_complete", "synthetic_pipeline_validated"):
            cases.append({"case": case["id"], "checked": False, "status": status.get("status")})
            continue
        cases.append(_recompute_case(
            case_dir,
            bars,
            initial,
            action_sources=action_sources,
            max_price_carry_sessions=max_price_carry_sessions,
        ))

    for bench_dir in sorted((run_dir / "benchmarks").glob("*")) if (run_dir / "benchmarks").exists() else []:
        if (bench_dir / "daily_account.csv").exists():
            cases.append(_recompute_case(
                bench_dir,
                bars,
                initial,
                action_sources=action_sources,
                max_price_carry_sessions=max_price_carry_sessions,
            ))

    checked = [c for c in cases if c.get("checked")]
    report = {
        "run_dir": str(run_dir),
        "protocol_id": registration.get("protocol_id"),
        "dataset_kind": registration.get("dataset_kind"),
        "synthetic": registration.get("synthetic"),
        "initial_capital": initial,
        "checked_cases": len(checked),
        "passed": sum(1 for c in checked if c["status"] == "passed"),
        "failed": sum(1 for c in checked if c["status"] == "failed"),
        "skipped": sum(1 for c in cases if not c.get("checked")),
        "cases": cases,
        "config_hashes": {
            "protocol": sha256_file(protocol_path),
        },
        "notes": [
            "对账只读取落盘成交/行动与原价, 不调用信号函数。",
            "金额差异容差 0.01 元, 份额容差 1e-8。",
        ],
    }
    if write:
        (run_dir / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="research.valuation_bands.verify")
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    report = verify_run(args.directory)
    print(json.dumps({k: report[k] for k in ("checked_cases", "passed", "failed", "skipped")},
                     ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
