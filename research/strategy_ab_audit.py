"""策略 A/B 的只读复核、亏损归因与周频诊断对照。

输出新目录；旧冻结实验、原始净值和生产指针均不修改。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import universe
from research.ex_nasdaq.core import market30
from research.goal30 import risk_overlay
from research.goal30.cash_replay import audit_cash_ledger, simulate_cash
from research.goal30.weekly_goal import weekly_overlay_allocator
from research.goal30.weekly_transfer import goal_checks
from src.market import load_market
from src.simulation import simulate

BASE = Path(__file__).resolve().parents[1]
A_DAILY = BASE / "output/goal30/asset_transfer/20260922-180211-a8de9f60"
A_WEEKLY = BASE / "output/goal30/weekly_transfer/20260922-180352-60dca6f9"
B_DAILY = BASE / "output/prod_candidate/runs/20260922-115635-1899ae31"
OUTPUT = BASE / "output/strategy_ab_audit"
FIRST_YEAR, LAST_YEAR = 2018, 2025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_report(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"净值日期重复或乱序：{path}")
    return frame


def calendar_metrics(report: pd.DataFrame, first_year=FIRST_YEAR, last_year=LAST_YEAR,
                     initial_capital=1_000_000.) -> dict:
    start = pd.Timestamp(f"{first_year}-01-01")
    end = pd.Timestamp(f"{last_year}-12-31")
    window = report.loc[start:end, "account"]
    prior = report.loc[report.index < start, "account"]
    base = float(prior.iloc[-1]) if len(prior) else float(initial_capital)
    if window.empty or window.index[0].year != first_year or window.index[-1].year != last_year:
        raise ValueError("共同日历窗口不完整")
    year_end = window.resample("YE").last()
    annual = year_end.pct_change(fill_method=None)
    annual.iloc[0] = year_end.iloc[0] / base - 1
    values = np.r_[base, window.to_numpy(dtype=float)]
    drawdown = 1 - values / np.maximum.accumulate(values)
    worst = annual.idxmin()
    return {
        "start": str(window.index[0].date()), "end": str(window.index[-1].date()),
        "years": last_year - first_year + 1,
        "cagr": float((window.iloc[-1] / base) ** (1 / (last_year - first_year + 1)) - 1),
        "max_drawdown": float(drawdown.max()),
        "worst_complete_year": {"year": int(worst.year), "return": float(annual.min())},
        "annual_returns": {str(day.year): float(value) for day, value in annual.items()},
    }


def audit_a_weekly() -> dict:
    completion = read_json(A_WEEKLY / "completion.json")
    registration = read_json(A_WEEKLY / "registration.json")
    archived = read_json(A_WEEKLY / "results.json")
    if sha256(A_WEEKLY / "results.json") != completion["results_sha256"]:
        raise ValueError("A 周频结果哈希不匹配")
    if sha256(A_WEEKLY / "registration.json") != completion["registration_sha256"]:
        raise ValueError("A 周频登记哈希不匹配")
    if len(archived["cases"]) != 24:
        raise ValueError("A 周频情景缺失")
    for relative, expected in registration["data_input_hashes"].items():
        if sha256(BASE / relative) != expected:
            raise ValueError(f"A 冻结行情输入变化：{relative}")

    decisions_count = set()
    all_joint = {}
    for name, row in archived["cases"].items():
        path = A_WEEKLY / f"report_{name}.csv"
        report = read_report(path)
        years = report.account.resample("YE").last()
        annual = years.pct_change(fill_method=None)
        annual.iloc[0] = years.iloc[0] / registration["policy"]["initial_capital"] - 1
        complete = annual[(annual.index.year >= 2017) & (annual.index.year <= 2025)]
        nav = report.account.to_numpy(dtype=float)
        actual = {
            "cagr": float((nav[-1] / nav[0]) ** (365.25 / (report.index[-1] - report.index[0]).days) - 1),
            "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
            "worst_complete_year": {"year": int(complete.idxmin().year),
                                    "return": float(complete.min())},
        }
        full = row["full_window"]
        for key in ("cagr", "max_drawdown"):
            if abs(actual[key] - full[key]) > 1e-10:
                raise ValueError(f"A 周频指标复算不一致：{name}/{key}")
        if actual["worst_complete_year"]["year"] != full["worst_complete_year"]["year"] or abs(
                actual["worst_complete_year"]["return"] - full["worst_complete_year"]["return"]) > 1e-10:
            raise ValueError(f"A 周频年度收益复算不一致：{name}")
        if row["ledger"]["violations"] or row["ledger"]["max_nav_error"] > .01:
            raise ValueError(f"A 周频现金账本未通过：{name}")
        signals = pd.read_csv(A_WEEKLY / f"signals_{name}.csv", parse_dates=["date"])
        trades = pd.read_csv(A_WEEKLY / f"trades_{name}.csv")
        weeks = report.index.to_period("W-SUN")
        allowed = set(report.index[np.r_[True, weeks[1:] != weeks[:-1]]])
        decision_days = set(signals.date)
        if decision_days != allowed - {report.index[-1]} and decision_days != allowed:
            raise ValueError(f"A 周频决策日不符：{name}")
        for trade in trades.itertuples(index=False):
            signal = pd.Timestamp(trade.signal_date)
            execution = pd.Timestamp(trade.date)
            if signal not in decision_days or report.index[report.index.get_loc(signal) + 1] != execution:
                raise ValueError(f"A 周频成交时序不符：{name}")
        decisions_count.add(len(signals))
        all_joint[name] = goal_checks(actual, registration["protocol"]["goals"])["joint_met"]

    primary = "communication60_gold35__weekly_trend200_vol15__base"
    candidate = registration["protocol"]["candidates"][0]
    variant = next(item for item in registration["protocol"]["variants"]
                   if item["name"] == "weekly_trend200_vol15")
    market = market30(BASE / registration["source"])
    trace = []
    selector = weekly_overlay_allocator(market.wide("adj_close"), variant, trace)
    policy = risk_overlay.risk_policy(registration["policy"], variant)
    with patch("research.goal30.cash_replay.allocator", selector), \
         patch("research.goal30.cash_replay.plan_orders", selector.plan):
        rerun, rerun_trades, rerun_actions, _, _ = simulate_cash(
            market, candidate, policy, registration["protocol"]["start"], registration["actions"])
    original_report = read_report(A_WEEKLY / f"report_{primary}.csv")
    if not rerun.index.equals(original_report.index):
        raise ValueError("A 主周频情景重新回放日期不同")
    replay_difference = float((rerun.account - original_report.account).abs().max())
    if replay_difference > .001:
        raise ValueError(f"A 主周频情景重新回放净值差异：{replay_difference}")
    original_trades = pd.read_csv(A_WEEKLY / f"trades_{primary}.csv")
    keys = ["date", "signal_date", "symbol", "side", "quantity"]
    if not rerun_trades[keys].reset_index(drop=True).equals(original_trades[keys].reset_index(drop=True)):
        raise ValueError("A 主周频情景重新回放成交不同")
    ledger = audit_cash_ledger(market, rerun, rerun_trades, rerun_actions,
                               policy["initial_capital"])
    return {"cases": len(archived["cases"]), "data_inputs_checked": len(registration["data_input_hashes"]),
            "source_differences_recorded": len(registration["stale_source_inputs"]),
            "decision_days_per_case": sorted(decisions_count),
            "joint_goal_cases": [name for name, value in all_joint.items() if value],
            "cash_ledger_and_next_session_checks": "passed",
            "primary_current_code_replay_max_nav_error_cny": replay_difference,
            "primary_current_code_replay_trades": len(rerun_trades),
            "primary_current_code_replay_ledger": ledger,
            "source_snapshot_limit": "旧冻结源码哈希与当前源码不一致；当前源码已对主周频情景逐笔重放，其余23例只读核对。"}


def a_2021_diagnostics() -> dict:
    suffix = "communication60_gold35__daily_trend200_vol15__base"
    weekly_suffix = "communication60_gold35__weekly_trend200_vol15__base"
    frames = {"daily": read_report(A_DAILY / f"report_{suffix}.csv"),
              "weekly": read_report(A_WEEKLY / f"report_{weekly_suffix}.csv")}
    result = {}
    month_values = {}
    for cadence, frame in frames.items():
        before = float(frame.loc[:"2020-12-31", "account"].iloc[-1])
        year = frame.loc["2021", :]
        month_values[cadence] = year.account.resample("ME").last() / before - 1
        holdings_path = ((A_DAILY if cadence == "daily" else A_WEEKLY) /
                         f"holdings_{suffix if cadence == 'daily' else weekly_suffix}.csv")
        holdings = pd.read_csv(holdings_path, parse_dates=["date"])
        values = holdings.pivot_table(index="date", columns="symbol", values="value", aggfunc="sum")
        weights = values.reindex(year.index).fillna(0).div(year.account, axis=0)
        trades_path = ((A_DAILY if cadence == "daily" else A_WEEKLY) /
                       f"trades_{suffix if cadence == 'daily' else weekly_suffix}.csv")
        trades = pd.read_csv(trades_path, parse_dates=["date"])
        year_trades = trades[trades.date.dt.year.eq(2021)]
        trades_by_day = {day: group for day, group in year_trades.groupby("date")}
        asset_pnl = defaultdict(float)
        commission = residual_actions = 0.
        previous_day = frame.index[frame.index.get_loc(year.index[0]) - 1]
        previous_nav = float(frame.at[previous_day, "account"])
        previous_values = values.reindex([previous_day]).fillna(0).iloc[0].to_dict()
        for day, row in year.iterrows():
            current_values = values.reindex([day]).fillna(0).iloc[0].to_dict()
            flows = defaultdict(float)
            fees = 0.
            for trade in trades_by_day.get(day, pd.DataFrame()).itertuples(index=False):
                sign = 1 if trade.side == "buy" else -1
                flows[trade.symbol] += sign * trade.quantity * trade.price
                fees += trade.commission
            daily_asset_pnl = 0.
            for symbol in set(previous_values) | set(current_values) | set(flows):
                gain = (current_values.get(symbol, 0.) - previous_values.get(symbol, 0.) -
                        flows.get(symbol, 0.))
                asset_pnl[symbol] += gain
                daily_asset_pnl += gain
            residual_actions += row.account - previous_nav - daily_asset_pnl + fees
            commission += fees
            previous_nav = row.account
            previous_values = current_values
        result[cadence] = {
            "annual_return": float(year.account.iloc[-1] / before - 1),
            "average_exposure_by_symbol": {key: float(weights.get(key, pd.Series(0., index=weights.index)).mean())
                                           for key in ("SH515880", "SH518880", "SH510300")},
            "trades": int(len(year_trades)), "sum_daily_turnover": float(year.turnover.sum()),
            "sum_daily_cost": float(year.cost.sum()),
            "asset_pnl_contribution_pp": {key: float(value / before * 100)
                                          for key, value in sorted(asset_pnl.items())},
            "commission_contribution_pp": float(-commission / before * 100),
            "corporate_action_and_other_residual_pp": float(residual_actions / before * 100),
            "monthly_average_exposure": {
                str(day.date()): {key: float(value) for key, value in row.items()}
                for day, row in weights.resample("ME").mean().fillna(0).iterrows()},
        }
    result["gap_weekly_minus_daily_pp"] = float(100 * (result["weekly"]["annual_return"] -
                                                   result["daily"]["annual_return"]))
    result["monthly_ytd_gap_pp"] = {
        str(day.date()): float(100 * (month_values["weekly"].loc[day] - month_values["daily"].loc[day]))
        for day in month_values["daily"].index
    }
    return result


def b_2018_attribution(market, name: str) -> dict:
    report = read_report(B_DAILY / f"report_{name}.csv").loc["2018"]
    trades = pd.read_csv(B_DAILY / f"trades_{name}.csv", parse_dates=["date"])
    trades = trades[trades.date.dt.year.eq(2018)]
    by_day = {day: part for day, part in trades.groupby("date")}
    factors = market.wide("adj_factor")
    adj = market.wide("adj_close").ffill(limit=5)
    units = defaultdict(float)
    cash = previous_nav = 1_000_000.
    contributions = defaultdict(float)
    costs = 0.
    max_error = 0.
    exposure = defaultdict(float)
    for day, row in report.iterrows():
        previous_day = market.calendar[market.calendar.get_loc(day) - 1]
        previous_values = {symbol: unit * adj.at[previous_day, symbol]
                           for symbol, unit in units.items() if unit > 1e-9}
        signed_flows = defaultdict(float)
        fees = 0.
        for trade in by_day.get(day, pd.DataFrame()).itertuples(index=False):
            sign = 1 if trade.side == "buy" else -1
            signed_flows[trade.symbol] += sign * trade.quantity * trade.price
            units[trade.symbol] += sign * trade.quantity / factors.at[day, trade.symbol]
            fees += trade.commission
            cash -= sign * trade.quantity * trade.price + trade.commission
        current_values = {symbol: unit * adj.at[day, symbol]
                          for symbol, unit in units.items() if unit > 1e-9}
        if abs(cash + sum(current_values.values()) - row.account) > max_error:
            max_error = abs(cash + sum(current_values.values()) - row.account)
        daily_gain = 0.
        for symbol in set(previous_values) | set(current_values) | set(signed_flows):
            gain = (current_values.get(symbol, 0.) - previous_values.get(symbol, 0.) -
                    signed_flows.get(symbol, 0.))
            contributions[symbol] += gain
            daily_gain += gain
        if abs(previous_nav + daily_gain - fees - row.account) > .01:
            raise ValueError(f"B {name} 2018年损益归因未平：{day.date()}")
        costs += fees
        previous_nav = row.account
        for symbol, value in current_values.items():
            exposure[symbol] += value / row.account / len(report)
    if max_error > .01:
        raise ValueError(f"B {name} 2018年持仓重建未平：{max_error}")
    result = {
        "annual_return": float(report.account.iloc[-1] / 1_000_000. - 1),
        "reconciliation_max_error_cny": float(max_error),
        "commission_contribution_pp": float(-costs / 1_000_000. * 100),
        "asset_contribution_pp": {symbol: float(gain / 1_000_000. * 100)
                                  for symbol, gain in sorted(contributions.items())},
        "average_exposure": {symbol: float(weight) for symbol, weight in sorted(exposure.items())},
        "trades": int(len(trades)),
    }
    result["asset_class_contribution_pp"] = {}
    for symbol, gain in contributions.items():
        asset_class = universe.get_asset_class(symbol[2:])
        result["asset_class_contribution_pp"][asset_class] = (
            result["asset_class_contribution_pp"].get(asset_class, 0.) + gain / 1_000_000. * 100)
    return result


def audit_b_and_run_weekly(directory: Path) -> tuple[dict, dict]:
    manifest = read_json(B_DAILY / "manifest.json")
    protocol = read_json(B_DAILY / "protocol.json")
    for relative, expected in manifest["engine"].items():
        if sha256(BASE / relative) != expected:
            raise ValueError(f"B 归档引擎源码已改变：{relative}")
    if sha256(BASE / "research/prod_candidate/protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("B 归档研究协议已改变")
    market = load_market(horizon=protocol["label_horizon"])
    if str(market.calendar[-1].date()) != manifest["data_end"]:
        raise ValueError("B 行情截止日改变")
    a_market = market30(BASE / "output/runs/20260912-220345-f20ea3f5")
    a_bars = a_market.bars.set_index(["date", "symbol"])
    b_bars = market.bars.set_index(["date", "symbol"])
    common = a_bars.index.intersection(b_bars.index)
    if len(common) != len(a_bars):
        raise ValueError("A 冻结行情不能全部匹配 B 本地行情")
    data_differences = {field: int((a_bars.loc[common, field] - b_bars.loc[common, field])
                                   .abs().gt(1e-8).sum())
                        for field in ("open", "close", "volume", "adj_factor")}
    if any(data_differences.values()):
        raise ValueError(f"A/B 同标的行情内容不同：{data_differences}")
    bond_eligibility = market.eligibility[market.eligibility.symbol.eq("SH511260")]
    bond_2018_eligible = int(bond_eligibility[bond_eligibility.date.dt.year.eq(2018)].eligible.sum())
    first_bond_eligible = str(bond_eligibility[bond_eligibility.eligible].date.min().date())
    first_sessions = set(pd.Series(market.calendar, index=market.calendar)
                         .groupby(market.calendar.to_period("W-SUN")).min())
    verification, weekly_cases, attributions = {}, {}, {}
    for name in ("lightgbm", "momentum"):
        scores = pd.read_pickle(B_DAILY / f"predictions_{name}.pkl")
        daily, daily_trades, daily_issues = simulate(market, scores, protocol)
        archived = read_report(B_DAILY / f"report_{name}.csv")
        if not daily.index.equals(archived.index):
            raise ValueError(f"B {name} 日频重放日期不同")
        difference = float((daily.account - archived.account).abs().max())
        if difference > .001:
            raise ValueError(f"B {name} 日频重放净值差异 {difference}")
        verification[name] = {"daily_replay_max_nav_error_cny": difference,
                              "daily_trade_count": len(daily_trades), "daily_issues": daily_issues}
        attributions[name] = b_2018_attribution(market, name)
        score_dates = scores.index.get_level_values("datetime")
        weekly_scores = scores.loc[score_dates.isin(first_sessions)]
        if weekly_scores.empty:
            raise ValueError(f"B {name} 周频信号为空")
        for slippage in (1., 2.):
            report, trades, issues = simulate(market, weekly_scores, protocol,
                                               slippage_multiplier=slippage)
            key = f"{name}_weekly_slippage_{slippage:g}"
            if not trades.empty:
                trade_signals = pd.to_datetime(trades.signal_date)
                trade_dates = pd.to_datetime(trades.date)
                if not trade_signals.isin(first_sessions).all():
                    raise ValueError(f"B {key} 有非周首交易日信号")
                next_dates = market.calendar[market.calendar.searchsorted(trade_signals) + 1]
                if not np.array_equal(next_dates.to_numpy(), trade_dates.to_numpy()):
                    raise ValueError(f"B {key} 非次交易日开盘成交")
            report.to_csv(directory / f"report_{key}.csv")
            trades.to_csv(directory / f"trades_{key}.csv", index=False)
            weekly_cases[key] = {
                "score_dates": int(weekly_scores.index.get_level_values("datetime").nunique()),
                "trades": int(len(trades)), "issues": issues,
                "common_2018_2025": calendar_metrics(report),
                "full_2018_2025_is_diagnostic": True,
            }
    return ({"engine_hashes_match": True, "protocol_hash_matches": True,
             "market_fingerprint": market.fingerprint,
             "market_overlap_rows": len(common), "market_field_differences": data_differences,
             "bond_511260_eligible_sessions_2018": bond_2018_eligible,
             "bond_511260_first_eligible": first_bond_eligible,
             "daily_replays": verification, "year_2018": attributions}, weekly_cases)


def run() -> Path:
    directory = OUTPUT / (pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d-%H%M%S") +
                          "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    a_audit = audit_a_weekly()
    a_2021 = a_2021_diagnostics()
    b_audit, b_weekly = audit_b_and_run_weekly(directory)
    a_weekly_results = read_json(A_WEEKLY / "results.json")["cases"]
    a_cases = {}
    for name in ("communication60_gold35__weekly_trend200_vol15__base",
                 "communication60_gold35__weekly_none__base",
                 "csi300_60_gold35__weekly_trend200_vol15__base"):
        a_cases[name] = calendar_metrics(read_report(A_WEEKLY / f"report_{name}.csv"))
    b_daily_cases = {name: calendar_metrics(read_report(B_DAILY / f"report_{name}.csv"))
                     for name in ("lightgbm", "momentum")}
    results = {
        "research_only": True,
        "goal": {"cagr_min": .135, "worst_complete_year_min": -.05,
                 "max_drawdown_limit": .40},
        "verification": {"A_weekly": a_audit, "B_and_market": b_audit},
        "loss_attribution": {"A_2021": a_2021, "B_2018": b_audit["year_2018"]},
        "comparison": {"A_weekly_cash_account": a_cases,
                       "B_daily_adjusted_total_return_proxy": b_daily_cases,
                       "B_weekly_adjusted_total_return_proxy": b_weekly,
                       "strict_account_comparability": False,
                       "reason": "A逐笔现金分红与应收账本；B复权因子全收益代理。B缺29只历史公司行动现金账本，不能声称同账户。"},
        "source_paths": {"A_daily": str(A_DAILY.relative_to(BASE)),
                         "A_weekly": str(A_WEEKLY.relative_to(BASE)),
                         "B_daily": str(B_DAILY.relative_to(BASE))},
    }
    (directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    sources = [Path(__file__), BASE / "research/goal30/weekly_transfer.py",
               BASE / "research/goal30/weekly_goal.py", BASE / "src/simulation.py",
               BASE / "src/portfolio.py", BASE / "src/market.py",
               A_WEEKLY / "results.json", A_DAILY / "results.json",
               B_DAILY / "manifest.json", B_DAILY / "protocol.json",
               B_DAILY / "predictions_lightgbm.pkl", B_DAILY / "predictions_momentum.pkl"]
    completion = {"completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                  "results_sha256": sha256(directory / "results.json"),
                  "inputs": {str(path.relative_to(BASE)).replace("\\", "/"): sha256(path)
                             for path in sources}}
    (directory / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"directory": str(directory), "A_joint_cases": a_audit["joint_goal_cases"],
                      "A_2021_gap_pp": a_2021["gap_weekly_minus_daily_pp"],
                      "B_weekly": {key: value["common_2018_2025"] for key, value in b_weekly.items()}},
                     ensure_ascii=False), flush=True)
    return directory


if __name__ == "__main__":
    run()
