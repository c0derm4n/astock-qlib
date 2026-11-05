"""Register and run the fixed MA breakout research comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from research.ma_breakout.engine import simulate_research_cash
from research.ma_breakout.signals import (
    BreakoutTargetSelector,
    SignalParameters,
    compute_technical_signals,
    effective_eligibility,
)
from src.artifacts import write_json_atomic
from src.formal_cash import audit_cash_ledger
from src.formal_rules import trend_targets, weekly_sessions
from src.market import load_market


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/ma_breakout"
RUNS = OUT / "runs"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
POINTER = ROOT / "output/strategies/current.json"
DATABASE = ROOT / "data/formal/market.duckdb"
REGISTRY = ROOT / "strategy_registry.json"
ACTIONS = ROOT / "strategy_corporate_actions.json"
ENGINE = Path(__file__).with_name("engine.py")
SIGNALS = Path(__file__).with_name("signals.py")
RUNNER = Path(__file__)

BASE_CASES = [
    "new_ma5_ma10_4pct_vol15__base",
    "new_ma5_ma10_4pct_no_vol__base",
    "formal_weekly_ma200_vol15__base",
    "daily_ma200_vol15__base",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_context():
    pointer = read_json(POINTER)
    registration_path = ROOT / pointer["directory"] / "registration.json"
    registration = read_json(registration_path)
    require(registration_path.parent.name == pointer["run_id"], "当前生产run指针不一致")
    require(digest(DATABASE) == registration["database_sha256"], "正式数据库哈希与registration不一致")
    require(pointer["data_end"] == "2026-10-19", "正式快照data_end不是2026-10-19")
    baseline = next(
        item for item in registration["registry"]["strategies"]
        if item["id"] == "trend200_vol15"
    )
    return pointer, registration_path, registration, baseline


def frozen_paths(registration_path):
    production = sorted((ROOT / "src").glob("formal_*.py"))
    return [
        POINTER,
        registration_path,
        DATABASE,
        REGISTRY,
        ACTIONS,
        PROTOCOL_PATH,
        ENGINE,
        SIGNALS,
        RUNNER,
        ROOT / "src/portfolio.py",
        ROOT / "src/market.py",
        ROOT / "src/universe_filter.py",
        registration_path.parent / "trend200_vol15_report.csv",
        registration_path.parent / "trend200_vol15_trades.csv",
        *production,
    ]


def snapshots(paths):
    return {relative(path): digest(path) for path in paths}


def validate_protocol(spec, pointer, registration, baseline):
    require(spec["start"] == registration["registry"]["start"], "实验起点与当前registry不同")
    require(spec["end"] == pointer["data_end"], "实验终点与当前正式快照不同")
    require(spec["initial_capital"] == registration["registry"]["initial_capital"], "初始资金不同")
    require(spec["source"]["baseline_strategy_id"] == baseline["id"], "基线策略ID不同")
    require(spec["allocation"]["communication"] == "SH515880", "通信标的改变")
    require(spec["allocation"]["gold"] == "SH518880", "黄金标的改变")
    require(spec["allocation"]["fallback"] == "SH510300", "资格替代标的改变")
    require(spec["technical_signal"]["threshold_base"] == .04, "基准阈值不是4%")
    require(spec["technical_signal"]["ma_entry_base"] == 5, "入场均线不是MA5")
    require(spec["technical_signal"]["ma_exit_base"] == 10, "退出均线不是MA10")
    require(spec["execution"]["base_delay_sessions"] == 1, "基础执行不是T+1")
    require(spec["execution"]["delay_stress_sessions"] == 2, "延迟压力不是T+2")
    require(spec["execution"]["tail_close_execution_tested"] is False, "本协议不能声称尾盘成交")


def register():
    spec = read_json(PROTOCOL_PATH)
    pointer, registration_path, registration, baseline = source_context()
    validate_protocol(spec, pointer, registration, baseline)
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    directory = RUNS / (
        pd.Timestamp.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=False)
    paths = frozen_paths(registration_path)
    hashes = snapshots(paths)
    payload = {
        "schema_version": 1,
        "status": "registered",
        "registered_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protocol": spec,
        "protocol_sha256": digest(PROTOCOL_PATH),
        "current_run_id": pointer["run_id"],
        "current_registry": registration["registry"],
        "market_fingerprint": registration["market_fingerprint"],
        "database_sha256": registration["database_sha256"],
        "baseline_candidate": copy.deepcopy(baseline["candidate"]),
        "baseline_policy": copy.deepcopy(baseline["policy"]),
        "actions_path": relative(ACTIONS),
        "input_hashes": hashes,
        "production_pointer_before": {
            "path": relative(POINTER),
            "sha256": digest(POINTER),
            "content": pointer,
        },
        "case_names": expected_case_names(),
    }
    write_json_atomic(directory / "registration.json", payload)
    registered = directory / "registered_code"
    registered.mkdir()
    for path in (PROTOCOL_PATH, ENGINE, SIGNALS, RUNNER):
        shutil.copy2(path, registered / path.name)
    write_json_atomic(OUT / "latest_registration.json", {
        "directory": relative(directory),
        "registration_sha256": digest(directory / "registration.json"),
    })
    print(json.dumps({
        "registered": relative(directory),
        "cases": len(payload["case_names"]),
    }, ensure_ascii=False))
    return directory


def verify_inputs(registration):
    changed = [
        path for path, expected in registration["input_hashes"].items()
        if not (ROOT / path).is_file() or digest(ROOT / path) != expected
    ]
    require(not changed, f"登记后输入发生变化：{changed}")
    current = read_json(POINTER)
    require(
        current == registration["production_pointer_before"]["content"]
        and digest(POINTER) == registration["production_pointer_before"]["sha256"],
        "生产指针在登记或运行期间发生变化",
    )


def scenario(name):
    if name == "base":
        return {"slippage_multiplier": 1., "execution_delay_sessions": 1}
    if name == "slippage_2":
        return {"slippage_multiplier": 2., "execution_delay_sessions": 1}
    if name == "delay_2":
        return {"slippage_multiplier": 1., "execution_delay_sessions": 2}
    raise ValueError(f"未知场景：{name}")


def expected_case_names():
    names = [
        "new_ma5_ma10_4pct_vol15__base",
        "new_ma5_ma10_4pct_vol15__slippage_2",
        "new_ma5_ma10_4pct_vol15__delay_2",
        "new_ma5_ma10_4pct_no_vol__base",
        "formal_weekly_ma200_vol15__base",
        "formal_weekly_ma200_vol15__slippage_2",
        "formal_weekly_ma200_vol15__delay_2",
        "daily_ma200_vol15__base",
        "daily_ma200_vol15__slippage_2",
        "daily_ma200_vol15__delay_2",
        "new_threshold_3pct__base",
        "new_threshold_5pct__base",
        "new_exit_ma8__base",
        "new_exit_ma12__base",
        "new_entry_ma4__base",
        "new_entry_ma6__base",
    ]
    require(len(names) == len(set(names)) == 16, "案例定义不是16个唯一场景")
    return names


def baseline_selector(candidate, policy, prices):
    trace = []

    def select(scores, allowed, current, holding_days, returns, current_policy, *, exiting=()):
        date = pd.Timestamp(returns.index[-1])
        targets = trend_targets(
            candidate, date, allowed, current, returns, prices,
            current_policy, exiting,
        )
        trace.append({
            "date": str(date.date()),
            "target_weights": json.dumps(targets, sort_keys=True),
            "eligible": json.dumps(sorted(allowed)),
            "exiting": json.dumps(sorted(exiting)),
        })
        return targets

    return select, trace


def build_breakout_context(market, candidate, start, parameters):
    communication = "SH515880"
    fallback = "SH510300"
    gold = "SH518880"
    prices = market.wide("adj_close")[[communication, fallback, gold]]
    eligibility = market.eligibility[
        market.eligibility.symbol.isin([communication, fallback, gold])
    ].pivot(index="date", columns="symbol", values="eligible").reindex(
        index=market.calendar, columns=[communication, fallback, gold]
    ).fillna(False).astype(bool)
    eligibility = effective_eligibility(
        eligibility, communication, fallback
    )
    book, frames = compute_technical_signals(
        prices, eligibility, parameters, state_start=start
    )
    return book, frames, prices


def flatten_signal_trace(trace):
    rows = []
    for row in trace:
        output = {
            "date": row["date"],
            "allowed": json.dumps(row["allowed"]),
            "exiting": json.dumps(row["exiting"]),
            "blocked_until_post_clear_entry": json.dumps(
                row.get("blocked_until_post_clear_entry", [])
            ),
            "raw_weights": json.dumps(row["raw_weights"], sort_keys=True),
            "target_weights": json.dumps(row["target_weights"], sort_keys=True),
        }
        for symbol, state in row.get("states", {}).items():
            for key, value in state.items():
                if isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                output[f"{symbol}.{key}"] = value
        rows.append(output)
    return pd.DataFrame(rows)


def opening_events(holdings, report_index):
    """Count zero-to-positive position transitions from end-of-day holdings."""
    if holdings.empty:
        return 0
    held = (
        holdings.pivot(index="date", columns="symbol", values="quantity")
        .reindex(pd.DatetimeIndex(report_index))
        .fillna(0.)
        .gt(1e-9)
    )
    transitions = held & ~held.shift(1, fill_value=False)
    return int(transitions.sum().sum())


def annual_table(nav, capital):
    ends = nav.groupby(nav.index.year).last()
    rows = []
    prior = float(capital)
    for year, end in ends.items():
        start_value = prior
        year_positions = np.flatnonzero(nav.index.year == year)
        first_position = int(year_positions[0])
        curve_values = (
            [start_value] + nav.iloc[first_position:year_positions[-1] + 1].tolist()
        )
        curve = pd.Series(curve_values, dtype=float)
        drawdown = 1 - curve / curve.cummax()
        rows.append({
            "year": int(year),
            "return": float(end / start_value - 1),
            "max_drawdown": float(drawdown.max()),
            "end_nav": float(end),
            "complete": bool(2017 <= year <= 2025),
        })
        prior = float(end)
    return rows


def window_metrics(nav, first_year, last_year):
    starts = np.flatnonzero(nav.index.year == first_year)
    ends = np.flatnonzero(nav.index.year == last_year)
    require(len(starts) and len(ends), f"净值缺少{first_year}-{last_year}窗口")
    first = int(starts[0])
    last = int(ends[-1])
    base = pd.Series(
        [nav.iloc[first - 1]], index=[nav.index[first - 1]]
    ) if first else pd.Series(dtype=float)
    curve = pd.concat([base, nav.iloc[first:last + 1]])
    drawdown = 1 - curve / curve.cummax()
    days = (curve.index[-1] - curve.index[0]).days
    rows = [
        row for row in annual_table(nav, nav.iloc[0])
        if first_year <= row["year"] <= last_year
    ]
    return {
        "start": str(curve.index[0].date()),
        "end": str(curve.index[-1].date()),
        "cumulative_return": float(curve.iloc[-1] / curve.iloc[0] - 1),
        "cagr": float((curve.iloc[-1] / curve.iloc[0]) ** (365.25 / days) - 1),
        "max_drawdown": float(drawdown.max()),
        "worst_complete_year": min(rows, key=lambda row: row["return"]) if rows else None,
    }


def metrics(report, trades, holdings, policy, *, state_frames=None):
    nav = report.account.astype(float)
    capital = float(policy["initial_capital"])
    drawdown = 1 - nav / nav.cummax()
    days = (nav.index[-1] - nav.index[0]).days
    annual = annual_table(nav, capital)
    complete = [row for row in annual if row["complete"]]
    require([row["year"] for row in complete] == list(range(2017, 2026)),
            "完整年度不是2017-2025")
    worst = min(complete, key=lambda row: row["return"])
    costs = float(
        (trades.commission + trades.slippage_cost.abs()).sum()
    ) if not trades.empty else 0.
    factors = {
        row["year"]: 1. + row["return"] for row in complete
    }
    excluding = math.prod(
        factors[year] for year in range(2017, 2025)
    )
    common_2020 = math.prod(
        factors[year] for year in range(2020, 2026)
    )
    stop_count = eligibility_exit_count = None
    technical_entries = technical_exits = None
    if state_frames is not None:
        state_start = pd.Timestamp(report.index[0])
        state_frames = {
            key: frame.loc[state_start:] for key, frame in state_frames.items()
        }
        previous_held = state_frames["held"].shift(1, fill_value=False)
        stop_count = int(
            (previous_held & state_frames["exit_condition"]).sum().sum()
        )
        eligibility_exit_count = int(
            (previous_held & ~state_frames["eligible"]).sum().sum()
        )
        technical_entries = {
            symbol: int(state_frames["entry_event"][symbol].sum())
            for symbol in state_frames["entry_event"].columns
        }
        technical_exits = {
            symbol: int(state_frames["exit_trigger"][symbol].sum())
            for symbol in state_frames["exit_trigger"].columns
        }
    result = {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "observations": len(nav),
        "final_nav": float(nav.iloc[-1]),
        "cumulative_return": float(nav.iloc[-1] / capital - 1),
        "cagr": float((nav.iloc[-1] / capital) ** (365.25 / days) - 1),
        "max_daily_nav_drawdown": float(drawdown.max()),
        "annual_table": annual,
        "worst_complete_year": worst,
        "trade_count": int(len(trades)),
        "opening_event_count": opening_events(holdings, nav.index),
        "stop_trigger_count": stop_count,
        "eligibility_exit_count": eligibility_exit_count,
        "technical_entry_events": technical_entries,
        "technical_exit_events": technical_exits,
        "average_position_weight": float((report.value / report.account).mean()),
        "average_cash_weight": float((report.cash / report.account).mean()),
        "total_turnover": float(report.turnover.sum()),
        "total_cost_fraction": float(report.cost.sum()),
        "turnover_cost_cny": costs,
        "subwindow_2017_2024": window_metrics(nav, 2017, 2024),
        "period_2017_2021": window_metrics(nav, 2017, 2021),
        "period_2022_2025": window_metrics(nav, 2022, 2025),
        "return_dependency_2025": {
            "factor_2025": float(factors[2025]),
            "factor_2017_2024_excluding_2025": float(excluding),
            "annualized_factor_excluding_2025": float(excluding ** (1 / 8) - 1),
            "diagnostic_only": "不构成删除2025后的连续回测或断裂序列回撤",
        },
        "communication_listing_2020_2025": {
            "compound_factor": float(common_2020),
            "cumulative_return": float(common_2020 - 1),
            "annualized_factor": float(common_2020 ** (1 / 6) - 1),
            "diagnostic": "通信上市后的共同完整年度收益，不用于另选参数",
        },
    }
    result["goal_checks"] = {
        "cagr": result["cagr"] >= .135,
        "worst_complete_year": worst["return"] >= -.05,
        "max_drawdown": result["max_daily_nav_drawdown"] <= .40,
    }
    result["joint_goal_met"] = all(result["goal_checks"].values())
    return result, drawdown


def baseline_parity(report, trades, registration):
    spec = registration["protocol"]["validation"]
    expected_rows = int(spec["expected_rows"])
    expected_trades = int(spec["expected_trades"])
    expected_final = float(spec["expected_final_nav"])
    path = ROOT / registration["protocol"]["source"]["baseline_report"]
    expected = pd.read_csv(path, index_col="date", parse_dates=True)
    require(report.index.equals(expected.index), "周频基线日期或行数不匹配")
    nav_error = float(np.max(np.abs(report.account.to_numpy() - expected.account.to_numpy())))
    result = {
        "expected_rows": expected_rows,
        "actual_rows": len(report),
        "expected_trades": expected_trades,
        "actual_trades": len(trades),
        "expected_final_nav": expected_final,
        "actual_final_nav": float(report.account.iloc[-1]),
        "max_nav_absolute_error": nav_error,
        "passed": bool(
            len(report) == expected_rows
            and len(trades) == expected_trades
            and abs(float(report.account.iloc[-1]) - expected_final) <= 1e-8
            and nav_error <= 1e-8
        ),
    }
    require(result["passed"], f"周频基线未对齐：{result}")
    return result


def case_definition(name, registration):
    parts = name.split("__")
    scenario_name = parts[-1]
    stem = "__".join(parts[:-1])
    parameters = SignalParameters()
    overlay = True
    strategy = "new"
    if stem == "new_ma5_ma10_4pct_vol15":
        pass
    elif stem == "new_ma5_ma10_4pct_no_vol":
        overlay = False
        require(scenario_name == "base", "no_vol只跑base")
    elif stem == "formal_weekly_ma200_vol15":
        strategy = "weekly"
    elif stem == "daily_ma200_vol15":
        strategy = "daily"
    elif stem == "new_threshold_3pct":
        parameters = SignalParameters(threshold=.03)
    elif stem == "new_threshold_5pct":
        parameters = SignalParameters(threshold=.05)
    elif stem == "new_exit_ma8":
        parameters = SignalParameters(exit_window=8)
    elif stem == "new_exit_ma12":
        parameters = SignalParameters(exit_window=12)
    elif stem == "new_entry_ma4":
        parameters = SignalParameters(entry_window=4)
    elif stem == "new_entry_ma6":
        parameters = SignalParameters(entry_window=6)
    else:
        raise ValueError(f"未知案例：{name}")
    return strategy, scenario(scenario_name), parameters, overlay


def run_case(directory, name, market, registration):
    spec = registration["protocol"]
    strategy, stress, parameters, overlay = case_definition(name, registration)
    candidate = copy.deepcopy(registration["baseline_candidate"])
    policy = copy.deepcopy(registration["baseline_policy"])
    policy["initial_capital"] = spec["initial_capital"]
    state_frames = None
    signal_trace = None
    if strategy in ("weekly", "daily"):
        prices = market.wide("adj_close")
        selector, trace = baseline_selector(candidate, policy, prices)
        signal_trace = pd.DataFrame(trace)
        dates = (
            weekly_sessions(market.calendar, spec["start"])
            if strategy == "weekly"
            else market.calendar[market.calendar >= pd.Timestamp(spec["start"])]
        )
    else:
        book, state_frames, _ = build_breakout_context(
            market, candidate, spec["start"], parameters
        )
        selector = BreakoutTargetSelector(
            book, candidate,
            communication=spec["allocation"]["communication"],
            fallback=spec["allocation"]["fallback"],
            gold=spec["allocation"]["gold"],
            volatility_target=policy["vol_target"] if overlay else None,
        )
        dates = market.calendar[market.calendar >= pd.Timestamp(spec["start"])]
    final_state = {}
    report, trades, actions, holdings, issues, order_log = simulate_research_cash(
        market, candidate, policy, spec["start"],
        read_json(ROOT / registration["actions_path"])["actions"],
        signal_dates=dates,
        execution_delay_sessions=stress["execution_delay_sessions"],
        slippage_multiplier=stress["slippage_multiplier"],
        selector=selector,
        state=final_state,
    )
    ledger = audit_cash_ledger(
        market, report, trades, actions, policy["initial_capital"]
    )
    if strategy == "new":
        signal_trace = flatten_signal_trace(selector.trace)
    parity = (
        baseline_parity(report, trades, registration)
        if name == "formal_weekly_ma200_vol15__base" else None
    )
    case_metrics, drawdown = metrics(
        report, trades, holdings, policy, state_frames=state_frames
    )
    case_metrics.update({
        "strategy": strategy,
        "parameters": parameters.__dict__,
        "scenario": stress,
        "volatility_overlay": overlay,
        "ledger": ledger,
        "issues": issues,
        "ending_cash": float(final_state["cash"]),
        "ending_positions": final_state["quantities"],
        "unexecuted_orders_at_end": int((order_log.status == "unexecuted_at_end").sum()),
        "unfilled_order_attempts": int((order_log.status == "unfilled").sum()),
        "liquidation_order_count": int(
            ((order_log.side == "sell") & (order_log.target_weight <= 1e-9)).sum()
        ),
        "baseline_parity": parity,
        "investment_admission": False,
    })
    for prefix, frame, index in (
        ("report", report, True),
        ("trades", trades, False),
        ("actions", actions, False),
        ("holdings", holdings, False),
        ("orders", order_log, False),
        ("signals", signal_trace, False),
    ):
        frame.to_csv(directory / f"{prefix}_{name}.csv", index=index)
    drawdown.rename("drawdown").to_csv(
        directory / f"drawdown_{name}.csv", index_label="date"
    )
    write_json_atomic(directory / f"case_{name}.json", case_metrics)
    artifacts = [
        path for path in directory.iterdir()
        if path.is_file() and path.name.endswith(f"_{name}.csv")
    ]
    case_metrics["artifact_hashes"] = {
        path.name: digest(path) for path in artifacts
    }
    write_json_atomic(directory / f"case_{name}.json", case_metrics)
    return case_metrics, report, drawdown


def plot_results(directory, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    candidates = [
        path for path in font_manager.findSystemFonts()
        if "msyh" in Path(path).name.lower() or "yahei" in Path(path).name.lower()
    ]
    if candidates:
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(
            fname=candidates[0]
        ).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    labels = {
        "new_ma5_ma10_4pct_vol15__base": "新 MA5/MA10 波动15%",
        "new_ma5_ma10_4pct_no_vol__base": "新 MA5/MA10 无波动缩放",
        "formal_weekly_ma200_vol15__base": "正式周频 MA200",
        "daily_ma200_vol15__base": "日频 MA200",
    }
    colors = dict(zip(labels, ["#0b6e69", "#d07a21", "#374151", "#b33a3a"]))
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for name, label in labels.items():
        if name not in reports:
            continue
        nav = reports[name].account / reports[name].account.iloc[0]
        axis.plot(nav.index, nav, label=label, color=colors[name], linewidth=1.5)
    axis.set_title("通信/黄金研究：基础情景累计净值")
    axis.set_xlabel("日期")
    axis.set_ylabel("起点归一化净值")
    axis.grid(alpha=.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "net_value.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6.5))
    for name, label in labels.items():
        if name not in reports:
            continue
        nav = reports[name].account
        drawdown = 1 - nav / nav.cummax()
        axis.plot(
            drawdown.index, drawdown, label=label,
            color=colors[name], linewidth=1.35,
        )
    axis.set_title("通信/黄金研究：每日净值回撤")
    axis.set_xlabel("日期")
    axis.set_ylabel("回撤幅度")
    axis.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    axis.grid(alpha=.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "drawdown.png", dpi=180)
    plt.close(fig)


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def money(value):
    return f"{value:,.2f}"


def write_report(directory, results, registration):
    cases = results["cases"]
    success = {
        name: row for name, row in cases.items()
        if row.get("status") == "completed"
    }
    base_names = [
        "new_ma5_ma10_4pct_vol15__base",
        "new_ma5_ma10_4pct_no_vol__base",
        "formal_weekly_ma200_vol15__base",
        "daily_ma200_vol15__base",
    ]
    display = {
        "new_ma5_ma10_4pct_vol15__base": "新 MA5/MA10 4%·波动15%",
        "new_ma5_ma10_4pct_no_vol__base": "新 MA5/MA10 4%·无波动缩放",
        "formal_weekly_ma200_vol15__base": "正式周频 MA200·波动15%",
        "daily_ma200_vol15__base": "日频 MA200·波动15%",
    }
    lines = [
        "# 通信/黄金 MA5-MA10 4% 日频研究",
        "",
        f"协议：`{registration['protocol']['version']}`；区间："
        f"{registration['protocol']['start']}—{registration['protocol']['end']}；"
        f"初始资金：{money(registration['protocol']['initial_capital'])} 元。",
        "",
        "本报告验证的是**收盘确认、下一交易日开盘执行**的可执行替代口径。"
        "没有分钟级数据，不能把结果解释为尾盘成交，也不能据此切换生产或获得实盘准入。",
        "",
        "主代理先做过一次主新内存诊断，v1和v2随后各完成完整16场景并向用户展示结果。"
        "v3只能标识为已查看v1/v2后的**执行语义修正复核**，不是策略首次查看结果前的预注册，"
        "也不是独立盲测。4%、MA5/MA10、仓位、波动预算、成本和全场景参数均未据结果调整。",
        "",
        "## 基础情景",
        "",
        "| 策略 | 期末资金 | CAGR | 最大每日净值回撤 | 最差完整年 | 成交笔数 | 开仓事件 | 平均仓位 | 平均现金 | 换手成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in base_names:
        row = success.get(name)
        if not row:
            continue
        worst = row["worst_complete_year"]
        lines.append(
            f"| {display[name]} | {money(row['final_nav'])} | {percent(row['cagr'])} | "
            f"{percent(row['max_daily_nav_drawdown'])} | "
            f"{worst['year']} {percent(worst['return'])} | {row['trade_count']} | "
            f"{row['opening_event_count']} | "
            f"{percent(row['average_position_weight'])} | "
            f"{percent(row['average_cash_weight'])} | {money(row['turnover_cost_cny'])} |"
        )
    lines += [
        "",
        "## 基线对齐",
        "",
    ]
    parity = success.get("formal_weekly_ma200_vol15__base", {}).get("baseline_parity")
    if parity:
        lines.append(
            f"正式周频基线逐日账户对齐通过：{parity['actual_rows']}行，"
            f"{parity['actual_trades']}笔成交，期末{money(parity['actual_final_nav'])}；"
            f"与正式CSV最大绝对差{parity['max_nav_absolute_error']:.3e}。"
        )
    lines += [
        "",
        "## 压力与敏感性",
        "",
        "| 案例 | 期末资金 | CAGR | 最大回撤 | 最差完整年 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in success.items():
        if name in BASE_CASES:
            continue
        worst = row["worst_complete_year"]
        lines.append(
            f"| `{name}` | {money(row['final_nav'])} | {percent(row['cagr'])} | "
            f"{percent(row['max_daily_nav_drawdown'])} | "
            f"{worst['year']} {percent(worst['return'])} |"
        )
    primary = success.get("new_ma5_ma10_4pct_vol15__base")
    if primary:
        lines += [
            "",
            "## 主方案年度",
            "",
            "| 年 | 收益 | 年内最大回撤 | 完整年 | 年末资金 |",
            "| --- | ---: | ---: | :---: | ---: |",
        ]
        for row in primary["annual_table"]:
            lines.append(
                f"| {row['year']} | {percent(row['return'])} | "
                f"{percent(row['max_drawdown'])} | {'是' if row['complete'] else '否'} | "
                f"{money(row['end_nav'])} |"
            )
        dependency = primary["return_dependency_2025"]
        listing = primary["communication_listing_2020_2025"]
        window = primary["subwindow_2017_2024"]
        entries = primary["technical_entry_events"]
        lines += [
            "",
            "## 期间诊断",
            "",
            f"- 技术入场事件：通信 {entries.get('SH515880', 0)} 次，"
            f"黄金 {entries.get('SH518880', 0)} 次，"
            f"沪深300替代 {entries.get('SH510300', 0)} 次。",
            f"- 2017—2024：CAGR {percent(window['cagr'])}，最大回撤 "
            f"{percent(window['max_drawdown'])}。",
            f"- 2025 收益因子：{dependency['factor_2025']:.6f}；"
            f"从2017—2025完整年度因子中仅剔除2025后，其余8年复合年化因子为 "
            f"{percent(dependency['annualized_factor_excluding_2025'])}。这是收益依赖诊断，"
            "不是删除2025后的连续回测，也不对断裂序列计算最大回撤。",
            f"- 2020—2025六个完整自然年通信上市后共同收益："
            f"{percent(listing['cumulative_return'])}，年化因子 "
            f"{percent(listing['annualized_factor'])}。该诊断不用于追加参数搜索。",
            f"- 2017—2021最大回撤 "
            f"{percent(primary['period_2017_2021']['max_drawdown'])}；"
            f"2022—2025最大回撤 "
            f"{percent(primary['period_2022_2025']['max_drawdown'])}。",
            "",
            "## 规则与限制",
            "",
            "- 技术价格使用各ETF自身复权收盘；1日/2日收益为价格比值，入场需完整两日历史和MA5，退出需完整MA10。",
            "- 同日退出优先；持仓无退出时延续，跌破MA5不单独卖出；不因缺少新4%事件清仓，也不金字塔加仓。",
            "- 通信只在其市场资格失效时使用合格沪深300替代；替代资产仅在其有效资格窗口累积自身MA5/MA10/4%状态。",
            "- 止损和资格失效从信号日锁定清仓意图；首次卖单失败后仍持续清仓，清零前不接受反弹entry。清零后的同一收盘只要当日entry_condition、资格满足且无exit，即可重新入场，不额外要求技术状态false→true。",
            "- T+2普通非零目标变化不撤销在途单；明确零目标撤销未成交买单，在途清仓卖单不撤销；拆分/除息撤单。",
            "- 所有历史结果均为已查看数据上的研究，不是独立样本外确认；失败场景全部保留，不按结果选赢家。",
            "- 本报告不改变生产策略、生产指针、正式旧冻结实验或数据。",
            "- v1 `20261023-110301-95f9e8f6`仅开仓事件统计受成交去重影响；"
            "v2 `20261023-111646-2e433f78`的清仓后解锁多要求了一次entry_event，"
            "导致T+2场景延迟压力结果有误。两目录均保留并由外部勘误记录说明。",
            "",
        ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(directory, quick=False):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(RUNS) and directory.is_dir(), "只能运行已登记研究目录")
    registration = read_json(directory / "registration.json")
    verify_inputs(registration)
    with (directory / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "quick": bool(quick),
        }, stream)
    started = time.monotonic()
    market = load_market(DATABASE, as_of=registration["protocol"]["end"])
    require(
        market.fingerprint == registration["market_fingerprint"],
        "加载后的market.fingerprint与正式registration不一致",
    )
    names = registration["case_names"]
    if quick:
        names = [name for name in names if name in BASE_CASES]
    results = {
        "status": "running",
        "quick": bool(quick),
        "investment_admission": False,
        "cases": {},
    }
    write_json_atomic(directory / "results.json", results)
    reports = {}
    for name in names:
        began = time.monotonic()
        try:
            case_result, report, _ = run_case(
                directory, name, market, registration
            )
            reports[name] = report
            results["cases"][name] = {
                "status": "completed",
                **case_result,
                "seconds": time.monotonic() - began,
            }
            print(json.dumps({
                "case": name,
                "status": "completed",
                "final_nav": case_result["final_nav"],
                "cagr": case_result["cagr"],
                "max_drawdown": case_result["max_daily_nav_drawdown"],
                "seconds": time.monotonic() - began,
            }, ensure_ascii=False), flush=True)
        except Exception as error:
            results["cases"][name] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - began,
            }
            print(json.dumps({
                "case": name,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }, ensure_ascii=False), flush=True)
        write_json_atomic(directory / "results.json", results)
    if reports:
        plot_results(directory, reports)
    results.update({
        "status": "completed_with_failures"
        if any(row["status"] == "failed" for row in results["cases"].values())
        else "completed",
        "completed_cases": sum(
            row["status"] == "completed" for row in results["cases"].values()
        ),
        "failed_cases": sum(
            row["status"] == "failed" for row in results["cases"].values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "production_pointer_after": {
            "path": relative(POINTER),
            "sha256": digest(POINTER),
            "content": read_json(POINTER),
        },
        "input_hashes_after": snapshots(
            [ROOT / path for path in registration["input_hashes"]]
        ),
    })
    write_json_atomic(directory / "results.json", results)
    write_report(directory, results, registration)
    verify_inputs(registration)
    require(
        results["production_pointer_after"] == {
            "path": relative(POINTER),
            "sha256": digest(POINTER),
            "content": read_json(POINTER),
        },
        "结果落盘后的生产指针与运行中快照不一致",
    )
    require(
        results["input_hashes_after"] == registration["input_hashes"],
        "结果落盘后的输入哈希发生变化",
    )
    artifacts = {
        path.name: digest(path) for path in directory.iterdir()
        if path.is_file() and path.name != "completion.json"
    }
    completion = {
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "status": results["status"],
        "quick": bool(quick),
        "expected_cases": len(names),
        "completed_cases": results["completed_cases"],
        "failed_cases": results["failed_cases"],
        "registration_sha256": digest(directory / "registration.json"),
        "results_sha256": digest(directory / "results.json"),
        "artifact_hashes": artifacts,
        "production_pointer_unchanged": True,
        "input_hashes_unchanged": True,
        "investment_admission": False,
    }
    write_json_atomic(directory / "completion.json", completion)
    write_json_atomic(OUT / "latest_completed.json", {
        "directory": relative(directory),
        "completion_sha256": digest(directory / "completion.json"),
    })
    print(json.dumps({
        "directory": relative(directory),
        "status": results["status"],
        "completed": results["completed_cases"],
        "failed": results["failed_cases"],
    }, ensure_ascii=False))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", type=Path)
    parser.add_argument(
        "--quick", action="store_true",
        help="只运行主方案、无波动辅助与两条基线的基础情景",
    )
    args = parser.parse_args()
    if args.register:
        require(not args.quick, "--quick只能与--run使用")
        register()
    else:
        run(args.run, quick=args.quick)


if __name__ == "__main__":
    main()
