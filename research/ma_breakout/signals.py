"""Causal MA5/MA10 breakout signals and target allocation for the research line."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.formal_rules import bounded_targets


@dataclass(frozen=True)
class SignalParameters:
    threshold: float = .04
    entry_window: int = 5
    exit_window: int = 10

    def validate(self):
        if (isinstance(self.entry_window, bool) or isinstance(self.exit_window, bool)
                or not isinstance(self.entry_window, int)
                or not isinstance(self.exit_window, int)
                or self.entry_window < 2 or self.exit_window < 2):
            raise ValueError("均线窗口必须是至少2的整数")
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("入场阈值必须是非负有限数")


def next_technical_state(held, eligible, entry_condition, exit_condition):
    """Advance one symbol/session with exits taking priority over entries."""
    if not eligible or exit_condition:
        return False
    if held:
        return True
    return bool(entry_condition)


def build_signal_conditions(prices, parameters=SignalParameters()):
    """Build causal entry/exit booleans from adjusted close prices."""
    parameters.validate()
    prices = pd.DataFrame(prices, dtype=float).sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("价格日期必须唯一且递增")
    usable = prices.where(np.isfinite(prices) & prices.gt(0))
    returns_1 = usable.pct_change(fill_method=None)
    returns_2 = usable / usable.shift(2) - 1
    ma_entry = usable.rolling(parameters.entry_window, min_periods=parameters.entry_window).mean()
    ma_exit = usable.rolling(parameters.exit_window, min_periods=parameters.exit_window).mean()
    complete_two_day_history = (
        usable.notna() & usable.shift(1).notna() & usable.shift(2).notna()
    )
    return_move = (
        (returns_1.ge(parameters.threshold) & np.isfinite(returns_1))
        | (returns_2.ge(parameters.threshold) & np.isfinite(returns_2))
    )
    entry_condition = (
        complete_two_day_history
        & np.isfinite(ma_entry)
        & usable.ge(ma_entry)
        & return_move
    )
    exit_condition = (
        usable.notna()
        & np.isfinite(ma_exit)
        & usable.lt(ma_exit)
    )
    return {
        "price": prices,
        "r1": returns_1,
        "r2": returns_2,
        "ma_entry": ma_entry,
        "ma_exit": ma_exit,
        "entry_condition": entry_condition,
        "exit_condition": exit_condition,
    }


def effective_eligibility(eligibility, communication, fallback):
    """Disable the fallback's technical state while communication is eligible."""
    result = pd.DataFrame(eligibility).copy().astype(bool)
    if communication not in result.columns or fallback not in result.columns:
        raise ValueError("资格矩阵缺少通信或替代标的")
    result[fallback] = result[fallback] & ~result[communication]
    return result


def compute_technical_signals(
    prices, eligibility, parameters=SignalParameters(), *, state_start=None
):
    """Apply the entry/exit latch without using any future observation."""
    conditions = build_signal_conditions(prices, parameters)
    prices = conditions["price"]
    eligible = pd.DataFrame(eligibility).reindex(
        index=prices.index, columns=prices.columns
    ).fillna(False).astype(bool)
    held = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entered = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exited = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    start = prices.index[0] if state_start is None else pd.Timestamp(state_start)
    if start not in prices.index:
        raise ValueError("技术状态起点必须是价格日历中的交易日")

    for row_index, date in enumerate(prices.index):
        if date < start:
            continue
        for symbol in prices.columns:
            before = held.iloc[row_index - 1][symbol] if row_index and date > start else False
            after = next_technical_state(
                bool(before),
                bool(eligible.at[date, symbol]),
                bool(conditions["entry_condition"].at[date, symbol]),
                bool(conditions["exit_condition"].at[date, symbol]),
            )
            held.at[date, symbol] = after
            entered.at[date, symbol] = after and not before
            exited.at[date, symbol] = bool(before) and not after

    frames = {
        **conditions,
        "eligible": eligible,
        "held": held,
        "entry_event": entered,
        "exit_trigger": exited,
    }
    columns = (
        "price", "r1", "r2", "ma_entry", "ma_exit", "eligible",
        "entry_condition", "exit_condition", "entry_event", "exit_trigger",
        "held",
    )
    trace = pd.concat(
        {name: frames[name].stack(future_stack=True) for name in columns},
        axis=1,
    )
    trace.index.names = ["date", "symbol"]
    return SignalBook(trace.sort_index()), frames


class SignalBook:
    """Date-indexed access to the precomputed causal state table."""

    def __init__(self, trace):
        trace = pd.DataFrame(trace).sort_index()
        required = {
            "eligible", "entry_condition", "exit_condition",
            "entry_event", "exit_trigger", "held",
        }
        if trace.empty or not required.issubset(trace.columns):
            raise ValueError("信号记录为空或缺少必要字段")
        if (not isinstance(trace.index, pd.MultiIndex)
                or list(trace.index.names) != ["date", "symbol"]
                or trace.index.has_duplicates):
            raise ValueError("信号记录必须使用唯一的date/symbol索引")
        self.trace = trace
        self._dates = {}
        for date, rows in trace.groupby(level="date", sort=False):
            self._dates[pd.Timestamp(date)] = rows.droplevel("date")

    def at(self, date):
        date = pd.Timestamp(date)
        if date not in self._dates:
            raise ValueError(f"信号日缺失：{date.date()}")
        return self._dates[date]


def bounded_daily_targets(desired, current, returns, policy):
    """Use the formal risk limiter, drift band, and shared turnover budget."""
    return pd.Series(
        bounded_targets(desired, current, returns, policy), dtype=float
    ).sort_index()


class BreakoutTargetSelector:
    """Stateful trace collector around a causal, precomputed SignalBook."""

    def __init__(
        self, book, candidate, *, communication, fallback, gold,
        volatility_target,
    ):
        if set(candidate.get("weights", {})) != {communication, gold}:
            raise ValueError("候选袖套必须严格为通信与黄金两个原始标的")
        if candidate.get("fallbacks", {}).get(communication) != {fallback: 1.}:
            raise ValueError("通信资格替代规则必须只指向沪深300")
        if volatility_target is not None and (
            not np.isfinite(volatility_target) or not 0 < volatility_target <= 1
        ):
            raise ValueError("波动目标必须位于0至1之间")
        self.book = book
        self.candidate = candidate
        self.communication = communication
        self.fallback = fallback
        self.gold = gold
        self.volatility_target = volatility_target
        self.trace = []
        self.blocked_until_post_clear_entry = set()

    def __call__(
        self, scores, allowed, current, holding_days, returns, policy, *, exiting=()
    ):
        date = pd.Timestamp(returns.index[-1])
        states = self.book.at(date)
        allowed = set(allowed)
        exiting = set(exiting)
        held_symbols = {
            symbol for symbol in (self.communication, self.fallback, self.gold)
            if abs(float(current.get(symbol, 0.))) > 1e-9
        }
        for symbol in held_symbols:
            state = states.loc[symbol]
            if symbol in exiting or bool(state["exit_trigger"]) or not bool(state["eligible"]):
                self.blocked_until_post_clear_entry.add(symbol)
        for symbol in list(self.blocked_until_post_clear_entry):
            state = states.loc[symbol]
            if (
                symbol not in held_symbols
                and symbol not in exiting
                and bool(state["eligible"])
                and bool(state["entry_condition"])
                and not bool(state["exit_condition"])
            ):
                self.blocked_until_post_clear_entry.remove(symbol)
        raw = {}
        communication_eligible = self.communication in allowed
        if communication_eligible:
            if bool(states.at[self.communication, "held"]) \
                    and self.communication not in exiting \
                    and self.communication not in self.blocked_until_post_clear_entry:
                raw[self.communication] = self.candidate["weights"][self.communication]
        elif self.fallback in allowed and bool(states.at[self.fallback, "held"]) \
                and self.fallback not in exiting \
                and self.fallback not in self.blocked_until_post_clear_entry:
            raw[self.fallback] = self.candidate["weights"][self.communication]
        if self.gold in allowed and bool(states.at[self.gold, "held"]) \
                and self.gold not in exiting \
                and self.gold not in self.blocked_until_post_clear_entry:
            raw[self.gold] = self.candidate["weights"][self.gold]

        working_policy = dict(policy)
        if self.volatility_target is None:
            # Keep every formal weight, group, equity and gross constraint;
            # only make the volatility comparison non-binding.
            working_policy["vol_target"] = 1e9
            working_policy["volatility_policy"] = "observe_only"
        else:
            if "vol_target" in working_policy \
                    and abs(float(working_policy["vol_target"]) - self.volatility_target) > 1e-12:
                raise ValueError("波动目标与当前registry定义不一致")
            working_policy["vol_target"] = float(self.volatility_target)
        targets = bounded_daily_targets(raw, current, returns, working_policy)
        state_rows = {}
        for symbol in (self.communication, self.fallback, self.gold):
            if symbol in states.index:
                state_rows[symbol] = {
                    key: states.at[symbol, key]
                    for key in (
                        "price", "r1", "r2", "ma_entry", "ma_exit",
                        "eligible", "entry_condition", "exit_condition",
                        "entry_event", "exit_trigger", "held",
                    )
                }
        self.trace.append({
            "date": str(date.date()),
            "allowed": sorted(allowed),
            "exiting": sorted(exiting),
            "blocked_until_post_clear_entry": sorted(
                self.blocked_until_post_clear_entry
            ),
            "raw_weights": raw,
            "target_weights": targets.to_dict(),
            "states": state_rows,
        })
        return targets.to_dict()
