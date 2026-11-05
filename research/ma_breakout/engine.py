"""Daily-signal cash replay built from the formal weekly cash engine.

The accounting and execution block is intentionally copied from
``src.formal_cash.simulate_cash`` in this repository. The research-only
changes are:

* the signal calendar is supplied explicitly;
* execution occurs exactly one or two market sessions after the signal;
* all orders remain frozen while in flight and at most one order per symbol
  may be pending;
* a later signal either preserves an unchanged in-flight order or cancels it
  and replaces it when the target changes;
* every order outcome is retained in a separate order ledger.

The production module remains the source for candidate validation, order
planning, portfolio budgeting, and the independent cash-ledger audit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.formal_orders import plan_orders
from src.formal_rules import SYMBOLS, validate_candidate
from src.portfolio import buy_notional_budget


PRODUCTION_ISSUE_KEYS = (
    "suspended_or_missing",
    "price_limit",
    "capacity_clipped",
    "cash_clipped",
    "risk_clipped",
    "corporate_action_order_cancelled",
)

ORDER_COLUMNS = (
    "order_id",
    "symbol",
    "side",
    "signal_date",
    "execution_date",
    "reference_price",
    "adv",
    "notional_limit",
    "target_weight",
    "quantity",
    "filled_quantity",
    "unfilled_quantity",
    "status",
    "reason",
    "cancelled_date",
    "last_attempt_date",
    "fill_date",
    "fill_price",
    "commission",
    "participation",
)


def build_execution_schedule(calendar, start, signal_dates, execution_delay_sessions):
    """Map explicit signal sessions to their deterministic execution sessions."""
    dates = pd.DatetimeIndex(calendar)
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("交易日历必须非空、唯一且递增")
    if isinstance(execution_delay_sessions, bool) or not isinstance(execution_delay_sessions, int) \
            or execution_delay_sessions not in (1, 2):
        raise ValueError("execution_delay_sessions 仅支持整数1或2")
    begin = max(int(dates.searchsorted(pd.Timestamp(start))), 1)
    if begin >= len(dates):
        raise ValueError("回放起点没有可用交易日")
    signals = pd.DatetimeIndex(pd.to_datetime(signal_dates)).normalize()
    if signals.isna().any() or not signals.is_unique or not signals.is_monotonic_increasing:
        raise ValueError("signal_dates 必须非空值、唯一且递增")
    signal_indices = dates.get_indexer(signals)
    if len(signals) and (signal_indices < 0).any():
        missing = signals[signal_indices < 0]
        raise ValueError(f"signal_dates 包含非交易日：{list(missing.strftime('%Y-%m-%d'))}")
    if len(signals) and (signal_indices < begin).any():
        raise ValueError("signal_dates 不能早于回放起点或首个有效会话")
    execution_indices = signal_indices + execution_delay_sessions
    execution_dates = [
        dates[index] if index < len(dates) else pd.NaT
        for index in execution_indices
    ]
    return pd.DataFrame({
        "signal_date": signals,
        "signal_index": signal_indices.astype(int),
        "execution_date": execution_dates,
        "execution_index": execution_indices.astype(int),
    })


def _action_calendar(dates, actions, payment_delay_sessions):
    splits, record_events, ex_events, pay_events = {}, {}, {}, {}
    event_ids = set()
    for event in actions:
        event_id = event["event_id"]
        if event_id in event_ids or event["symbol"] not in SYMBOLS:
            raise ValueError("公司行动ID重复或标的不在30只池中")
        event_ids.add(event_id)
        if event["type"] == "split":
            splits.setdefault(pd.Timestamp(event["effective_date"]), []).append(event)
        else:
            record_events.setdefault(pd.Timestamp(event["record_date"]), []).append(event)
            ex_events.setdefault(pd.Timestamp(event["ex_date"]), []).append(event)
            payment_index = int(dates.searchsorted(pd.Timestamp(event["payment_date"]))) + payment_delay_sessions
            if payment_index < len(dates):
                pay_events.setdefault(dates[payment_index], []).append(event)
    return splits, record_events, ex_events, pay_events


def _new_order_row(order, order_id, signal_date, execution_date):
    raw_quantity = float(order["quantity"])
    return {
        "order_id": order_id,
        "symbol": order["symbol"],
        "side": order["side"],
        "signal_date": str(signal_date.date()),
        "execution_date": None if pd.isna(execution_date) else str(pd.Timestamp(execution_date).date()),
        "reference_price": float(order["reference_price"]),
        "adv": float(order["adv"]),
        "notional_limit": float(order["notional_limit"]),
        "target_weight": float(order["target_weight"]),
        "quantity": raw_quantity,
        "filled_quantity": 0.,
        "unfilled_quantity": raw_quantity,
        "status": "scheduled",
        "reason": None,
        "cancelled_date": None,
        "last_attempt_date": None,
        "fill_date": None,
        "fill_price": None,
        "commission": 0.,
        "participation": None,
        "_execution_index": int(order["_execution_index"]),
        "_full_exit": bool(order.get("full_exit", False)),
    }


def _finish_order(row, status, reason, *, filled=0., date=None, fill_price=None,
                  commission=0., participation=None):
    row["status"] = status
    row["reason"] = reason
    row["filled_quantity"] = float(filled)
    row["unfilled_quantity"] = max(0., float(row["quantity"]) - float(filled))
    if date is not None:
        row["last_attempt_date"] = str(pd.Timestamp(date).date())
    if date is not None and filled > 1e-12:
        row["fill_date"] = str(pd.Timestamp(date).date())
    if fill_price is not None:
        row["fill_price"] = float(fill_price)
    row["commission"] = float(commission)
    row["participation"] = None if participation is None else float(participation)


def _cancel_order(row, status, reason, date):
    row["status"] = status
    row["reason"] = reason
    row["cancelled_date"] = str(pd.Timestamp(date).date())
    row["unfilled_quantity"] = float(row["quantity"])


def simulate_research_cash(
    market,
    candidate,
    policy,
    start,
    actions,
    *,
    signal_dates,
    execution_delay_sessions,
    slippage_multiplier=1.,
    payment_delay_sessions=0,
    selector=None,
    observer=None,
    state=None,
    order_planner=None,
):
    """Replay a fixed target strategy with explicit daily signals.

    Returns ``report, trades, actions_log, holdings, issues, order_log``.
    ``report``, ``trades`` and ``actions_log`` retain the production cash
    engine's columns so ``src.formal_cash.audit_cash_ledger`` can be called
    directly without an adapter.
    """
    validate_candidate(candidate)
    if (candidate["family"] != "fixed" or isinstance(payment_delay_sessions, bool)
            or not isinstance(payment_delay_sessions, int) or payment_delay_sessions < 0
            or not np.isfinite(slippage_multiplier) or slippage_multiplier <= 0):
        raise ValueError("现金回放仅用于冻结固定权重候选，支付延迟必须非负")

    close = market.wide("close")
    opened = market.wide("open")
    volume = market.wide("volume")
    adjusted = market.wide("adj_close")
    close_mark = close.ffill(limit=5)
    returns = adjusted.pct_change(fill_method=None)
    eligibility = market.eligibility.pivot(index="date", columns="symbol", values="eligible")
    adv = market.eligibility.pivot(index="date", columns="symbol", values="avg_amount")
    dates = pd.DatetimeIndex(market.calendar)
    schedule = build_execution_schedule(dates, start, signal_dates, execution_delay_sessions)
    begin = max(int(dates.searchsorted(pd.Timestamp(start))), 1)
    signal_indices = set(schedule["signal_index"].astype(int))
    split_events, record_events, ex_events, pay_events = _action_calendar(
        dates, actions, payment_delay_sessions
    )

    cash = previous_nav = float(policy["initial_capital"])
    quantities, bought, entitled, receivable = {}, {}, {}, {}
    exiting, queue, records, trades, actions_log, holdings = set(), {}, [], [], [], []
    issues = {key: 0 for key in PRODUCTION_ISSUE_KEYS}
    issues.update({
        "in_flight_signal_suppressed": 0,
        "ordinary_target_changes_preserved": 0,
        "orders_superseded_by_signal": 0,
        "insufficient_position": 0,
        "unexecuted_at_end": 0,
    })
    order_rows, active_orders = [], []
    planner = order_planner or plan_orders

    for i in range(begin, len(dates)):
        date = dates[i]
        cost = turnover = 0.
        action_symbols = set()
        for event in split_events.get(date, []):
            symbol = event["symbol"]
            ratio = float(event["ratio"])
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("折算比例无效")
            before = quantities.get(symbol, 0.)
            if before:
                quantities[symbol] = before * ratio
            action_symbols.add(symbol)
            actions_log.append({
                "date": str(date.date()), "phase": "pre_open", "type": "split",
                "event_id": event["event_id"], "symbol": symbol, "ratio": ratio,
                "quantity_before": before, "quantity_after": before * ratio,
            })
        for event in ex_events.get(date, []):
            amount = entitled.get(event["event_id"], 0.)
            receivable[event["event_id"]] = amount
            action_symbols.add(event["symbol"])
            actions_log.append({
                "date": str(date.date()), "phase": "pre_open", "type": "accrue_dividend",
                "event_id": event["event_id"], "symbol": event["symbol"], "amount": amount,
            })

        # A split or ex-date invalidates every frozen order for that symbol,
        # including orders whose original execution date is later than today.
        for row in active_orders:
            if row["status"] == "scheduled" and row["symbol"] in action_symbols:
                _cancel_order(row, "cancelled_corporate_action", "corporate_action", date)
                issues["corporate_action_order_cancelled"] += 1

        for row in queue.pop(i, []):
            if row["status"] != "scheduled":
                continue
            symbol, side = row["symbol"], row["side"]
            price = opened.at[date, symbol]
            if not np.isfinite(price) or price <= 0 or volume.at[date, symbol] <= 0:
                _finish_order(row, "unfilled", "suspended_or_missing", date=date)
                issues["suspended_or_missing"] += 1
                continue
            threshold = .195 if symbol in ("SH588000", "SZ159915") else .095
            if abs(price / row["reference_price"] - 1) >= threshold:
                _finish_order(row, "unfilled", "price_limit", date=date)
                issues["price_limit"] += 1
                continue

            requested = math.floor(row["quantity"] / 100) * 100
            if side == "sell" and row["_full_exit"]:
                requested = quantities.get(symbol, 0.)
            if side == "sell" and row["target_weight"] == 0 and row["quantity"] < 100 and not row["_full_exit"]:
                requested = row["quantity"]
            capacity = math.floor(row["notional_limit"] / price / 100) * 100
            reasons = []
            quantity = requested
            if quantity > capacity:
                issues["capacity_clipped"] += 1
                reasons.append("capacity_clipped")
                quantity = capacity
            if side == "sell":
                available = quantities.get(symbol, 0.)
                if quantity > available:
                    issues["insufficient_position"] += 1
                    reasons.append("insufficient_position")
                    quantity = available
            if quantity <= 0:
                reason = "+".join(reasons) if reasons else "no_executable_quantity"
                _finish_order(row, "unfilled", reason, date=date)
                continue

            participation = quantity * price / max(row["adv"], 1.)
            slip = policy["slippage"] * slippage_multiplier \
                + policy["impact_coefficient"] * math.sqrt(participation)
            fill_price = price * (1 + slip if side == "buy" else 1 - slip)
            if side == "buy":
                if policy.get("conservative_buy_budget"):
                    prices = close_mark.iloc[i - 1].to_dict()
                    prices.update(opened.loc[date].dropna().to_dict())
                    room = buy_notional_budget(symbol, quantities, cash, prices, policy)
                    cap = math.floor(
                        room / (fill_price * (1 + policy["commission"])) / 100
                    ) * 100
                    if quantity > cap:
                        issues["risk_clipped"] += 1
                        reasons.append("risk_clipped")
                        quantity = cap
                affordable = math.floor(
                    max(0., cash - policy["minimum_commission"])
                    / (fill_price * (1 + policy["commission"])) / 100
                ) * 100
                if quantity > affordable:
                    issues["cash_clipped"] += 1
                    reasons.append("cash_clipped")
                    quantity = affordable
            if quantity <= 0:
                reason = "+".join(reasons) if reasons else "no_executable_quantity"
                _finish_order(row, "unfilled", reason, date=date)
                continue

            fee = max(policy["minimum_commission"], quantity * fill_price * policy["commission"])
            direction = 1 if side == "buy" else -1
            cash -= direction * quantity * fill_price + fee
            if side == "buy" and quantities.get(symbol, 0.) <= 1e-9:
                bought[symbol] = i
            quantities[symbol] = quantities.get(symbol, 0.) + direction * quantity
            if quantities[symbol] <= 1e-8:
                quantities.pop(symbol, None)
                bought.pop(symbol, None)
                exiting.discard(symbol)
            elif side == "sell" and row["target_weight"] <= 1e-9:
                exiting.add(symbol)
            cost += fee + quantity * price * slip
            turnover += quantity * price
            trades.append({
                "date": str(date.date()), "signal_date": row["signal_date"],
                "symbol": symbol, "side": side, "quantity": quantity,
                "price": fill_price, "commission": fee,
                "slippage_cost": quantity * price * slip,
                "participation": participation,
            })
            status = "filled" if abs(quantity - requested) <= 1e-8 else "partial"
            reason = "+".join(reasons) if reasons else None
            _finish_order(
                row, status, reason, filled=quantity, date=date,
                fill_price=fill_price, commission=fee, participation=participation,
            )

        for event in record_events.get(date, []):
            quantity = quantities.get(event["symbol"], 0.)
            amount = quantity * event["cash_per_share"]
            entitled[event["event_id"]] = amount
            actions_log.append({
                "date": str(date.date()), "phase": "post_close",
                "type": "record_entitlement", "event_id": event["event_id"],
                "symbol": event["symbol"], "quantity": quantity,
                "cash_per_share": event["cash_per_share"], "amount": amount,
            })
        for event in pay_events.get(date, []):
            amount = receivable.pop(event["event_id"], 0.)
            cash += amount
            actions_log.append({
                "date": str(date.date()), "phase": "post_close",
                "type": "pay_dividend", "event_id": event["event_id"],
                "symbol": event["symbol"], "amount": amount,
            })

        values = {}
        for symbol, quantity in quantities.items():
            price = close_mark.at[date, symbol]
            if not np.isfinite(price):
                raise ValueError("持仓连续缺价，不能虚构估值")
            values[symbol] = quantity * price
            holdings.append({
                "date": date, "symbol": symbol, "quantity": quantity,
                "value": values[symbol],
            })
        liquid_nav = cash + sum(values.values())
        claims = sum(receivable.values())
        nav = liquid_nav + claims
        if cash < -1e-6 or liquid_nav <= 0 or nav <= 0:
            raise ValueError("现金账户余额或净值无效")
        net = nav / previous_nav - 1
        records.append({
            "date": date, "account": nav,
            "return": net + cost / previous_nav,
            "cost": cost / previous_nav,
            "turnover": turnover / previous_nav,
            "cash": cash, "value": sum(values.values()),
            "dividend_receivable": claims,
        })

        if i in signal_indices:
            allowed = eligibility.columns[eligibility.loc[date].fillna(False)].tolist()
            current = {symbol: value / liquid_nav for symbol, value in values.items()}
            targets = selector(
                pd.Series(dtype=float),
                allowed,
                current,
                {symbol: i - bought[symbol] + 1 for symbol in quantities},
                returns.loc[:date],
                policy,
                exiting=exiting,
            ) if selector is not None else {}
            if targets is None:
                targets = {}
            targets = pd.Series(targets, dtype=float).fillna(0.).to_dict()

            pending_by_symbol = {}
            for row in active_orders:
                if row["status"] != "scheduled":
                    continue
                symbol = row["symbol"]
                if symbol in pending_by_symbol:
                    raise ValueError(f"{symbol} 存在多组在途委托")
                pending_by_symbol[symbol] = row

            retained_symbols = set()
            for symbol, row in list(pending_by_symbol.items()):
                new_target = float(targets.get(symbol, 0.))
                if row["side"] == "buy" and row["target_weight"] > 0 and new_target == 0:
                    _cancel_order(
                        row, "cancelled_by_stop_or_eligibility",
                        "explicit_zero_target",
                        date,
                    )
                    issues["orders_superseded_by_signal"] += 1
                    pending_by_symbol.pop(symbol)
                else:
                    retained_symbols.add(symbol)
                    if abs(new_target - row["target_weight"]) > 1e-10:
                        issues["ordinary_target_changes_preserved"] += 1
                    else:
                        issues["in_flight_signal_suppressed"] += 1

            planning_targets = dict(targets)
            for symbol in retained_symbols:
                planning_targets[symbol] = current.get(symbol, 0.)
            planned_orders = planner(
                planning_targets, quantities, cash, close_mark.loc[date], adv.loc[date], policy
            )
            new_orders = []
            execution_index = i + execution_delay_sessions
            execution_date = dates[execution_index] if execution_index < len(dates) else pd.NaT
            for order in planned_orders:
                if order["symbol"] in retained_symbols:
                    raise ValueError("订单规划器为已在途标的重复生成委托")
                enriched = dict(order)
                enriched.update({
                    "signal_date": str(date.date()),
                    "adv": float(adv.at[date, order["symbol"]]),
                    "_execution_index": execution_index,
                })
                row = _new_order_row(
                    enriched, f"{date:%Y%m%d}-{i:04d}-{len(order_rows):04d}",
                    date, execution_date,
                )
                order_rows.append(row)
                active_orders.append(row)
                new_orders.append(enriched)
                if execution_index < len(dates):
                    queue.setdefault(execution_index, []).append(row)
            if observer is not None:
                observer(
                    date, quantities.copy(), cash, bought.copy(), exiting.copy(),
                    targets, new_orders,
                )
        previous_nav = nav

    for row in active_orders:
        if row["status"] == "scheduled":
            _cancel_order(row, "unexecuted_at_end", "replay_end_before_execution", dates[-1])
            row["cancelled_date"] = None
            issues["unexecuted_at_end"] += 1

    report = pd.DataFrame(records).set_index("date")
    report["bench"] = (
        market.wide("adj_close")["SH510300"]
        .pct_change(fill_method=None)
        .reindex(report.index)
        .fillna(0.)
    )
    if state is not None:
        pending = [row for row in active_orders if row["status"] == "scheduled"]
        state.update(
            cash=cash,
            quantities=dict(quantities),
            bought=dict(bought),
            exiting=sorted(exiting),
            dividend_receivable=sum(receivable.values()),
            pending_orders=[dict(row) for row in pending],
        )
    for row in order_rows:
        row.pop("_execution_index", None)
        row.pop("_full_exit", None)
    order_log = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    return (
        report,
        pd.DataFrame(trades),
        pd.DataFrame(actions_log),
        pd.DataFrame(holdings),
        issues,
        order_log,
    )
