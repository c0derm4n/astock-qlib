"""正式估值账户与执行引擎: 真实现金账本、订单、公司行动、每日净值与指标。

从经审查的研究版本迁入；账户口径由本模块独立实现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibrationBook, CalibrationSnapshot
from .data import ResearchData, session_ts
from .signals import (
    AccountSnapshot,
    Decision,
    OrderIntent,
    SignalContext,
    StrategyState,
    evaluate_dividend_module,
    evaluate_weekly,
    ratio_gate_series,
)


# ------------------------------------------------------------------ 参数解析
@dataclass
class ResolvedCase:
    case_id: str
    group: str
    strategy: str
    kind: str
    index_id: str
    metric: str
    symbol: str
    w_cap: float
    exit_q: float
    band_upper_mult: float
    cooldown_sessions: int
    delay_sessions: int
    slippage: float
    fixed_exit_level: Optional[float]
    gate: Optional[str]
    dividend_module: bool
    raw: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "group": self.group,
            "strategy": self.strategy,
            "kind": self.kind,
            "index_id": self.index_id,
            "metric": self.metric,
            "symbol": self.symbol,
            "w_cap": self.w_cap,
            "exit_q": self.exit_q,
            "band_upper_mult": self.band_upper_mult,
            "cooldown_sessions": self.cooldown_sessions,
            "delay_sessions": self.delay_sessions,
            "slippage": self.slippage,
            "fixed_exit_level": self.fixed_exit_level,
            "gate": self.gate,
            "dividend_module": self.dividend_module,
        }


def resolve_case(protocol: Mapping[str, object], case: Mapping[str, object]) -> ResolvedCase:
    strategy = str(case["strategy"])
    base = dict(protocol["strategies"][strategy])
    overrides = dict(case.get("overrides") or {})
    merged = {**base, **overrides}
    return ResolvedCase(
        case_id=str(case["id"]),
        group=str(case.get("group", "primary")),
        strategy=strategy,
        kind=str(case.get("kind", "primary")),
        index_id=str(merged["index_id"]),
        metric=str(merged["metric"]),
        symbol=str(merged["etf"]),
        w_cap=float(merged["w_cap"]),
        exit_q=float(merged["exit_q"]),
        band_upper_mult=float(merged.get("band_upper_mult", protocol["calibration"]["band_upper_mult"])),
        cooldown_sessions=int(merged.get("cooldown_sessions", 20)),
        delay_sessions=int(merged.get("delay_sessions", 1)),
        slippage=float(merged.get("slippage", protocol["execution"]["slippage"])),
        fixed_exit_level=(float(merged["fixed_exit_level"]) if merged.get("fixed_exit_level") is not None else None),
        gate=(str(merged["gate"]) if merged.get("gate") else None),
        dividend_module=bool(merged.get("dividend_module", False)),
        raw=merged,
    )


def calibration_params(protocol: Mapping[str, object], case: ResolvedCase) -> Dict[str, object]:
    params = dict(protocol["calibration"])
    params["band_upper_mult"] = case.band_upper_mult
    return params


# ------------------------------------------------------------------ 账户
@dataclass
class ReceivableBucket:
    event_id: str
    symbol: str
    amount: float
    payment_date: Optional[pd.Timestamp]


class Portfolio:
    """真实现金账本: 现金、份额、分红应收与费用。"""

    def __init__(self, symbols: Sequence[str], initial_capital: float):
        self.symbols = list(symbols)
        self.cash = float(initial_capital)
        self.quantity: Dict[str, int] = {s: 0 for s in self.symbols}
        self.cost_basis: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.receivables: List[ReceivableBucket] = []
        self.realized_pnl: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.dividend_income: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.total_fees = 0.0
        self.total_turnover = 0.0
        self.bought_today: Dict[str, int] = {s: 0 for s in self.symbols}

    @property
    def receivable(self) -> float:
        return float(sum(b.amount for b in self.receivables))

    def value(self, symbol: str, prices: Mapping[str, float]) -> float:
        return self.quantity.get(symbol, 0) * float(prices.get(symbol, 0.0))

    def market_value(self, prices: Mapping[str, float]) -> float:
        return float(sum(self.value(s, prices) for s in self.symbols))

    def sellable_quantity(self, symbol: str) -> int:
        return max(
            0,
            int(self.quantity.get(symbol, 0)) - int(self.bought_today.get(symbol, 0)),
        )

    def nav(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.market_value(prices) + self.receivable

    def add_receivable(self, bucket: ReceivableBucket):
        self.receivables.append(bucket)

    def settle_receivables(self, session: pd.Timestamp) -> List[Dict[str, object]]:
        settled = []
        remaining = []
        for bucket in self.receivables:
            if bucket.payment_date is not None and bucket.payment_date <= session:
                self.cash += bucket.amount
                self.dividend_income[bucket.symbol] = self.dividend_income.get(bucket.symbol, 0.0) + bucket.amount
                settled.append({"event_id": bucket.event_id, "symbol": bucket.symbol,
                                "amount": bucket.amount, "paid_session": str(pd.Timestamp(session).date())})
            else:
                remaining.append(bucket)
        self.receivables = remaining
        return settled


# ------------------------------------------------------------------ 执行
@dataclass
class PendingOrder:
    order_id: str
    signal_session: pd.Timestamp
    exec_session: pd.Timestamp
    side: str
    symbol: str
    reason: str
    requested_qty: Optional[int]
    reference_price: float
    adv20: Optional[float]
    target_weight: Optional[float] = None
    requires_symbol_flat: Optional[str] = None
    status: str = "pending"
    filled_qty: int = 0
    fill_price: Optional[float] = None
    commission: float = 0.0
    impact: float = 0.0
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        return {
            "order_id": self.order_id,
            "signal_date": str(pd.Timestamp(self.signal_session).date()),
            "signal_timestamp": str(session_ts(self.signal_session)),
            "execution_date": str(pd.Timestamp(self.exec_session).date()),
            "side": self.side,
            "symbol": self.symbol,
            "reason": self.reason,
            "requested_quantity": self.requested_qty,
            "filled_quantity": self.filled_qty,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "adv20_snapshot": self.adv20,
            "commission": self.commission,
            "impact": self.impact,
            "status": self.status,
            "requires_symbol_flat": self.requires_symbol_flat,
            "note": self.note,
        }


def _impact(notional: float, adv20: Optional[float], coef: float) -> float:
    if not adv20 or adv20 <= 0 or notional <= 0:
        return 0.0
    return float(coef * math.sqrt(notional / adv20))


def _commission(notional: float, rate: float, minimum: float) -> float:
    if notional <= 0:
        return 0.0
    return float(max(minimum, notional * rate))


def _max_buy_lots_with_cash_floor(
    cash: float,
    nav_open: float,
    price: float,
    min_cash_share: float,
    lot_size: int,
    commission_rate: float,
    min_commission: float,
) -> int:
    """返回交易费后仍不跌破现金下限的最大整手买量。"""
    if price <= 0 or lot_size <= 0:
        return 0
    spendable = max(0.0, cash - min_cash_share * nav_open)
    lots = int(spendable / price // lot_size) * lot_size
    while lots > 0:
        notional = lots * price
        if notional + _commission(notional, commission_rate, min_commission) <= spendable + 1e-9:
            return lots
        lots -= lot_size
    return 0


# ------------------------------------------------------------------ 回放主体
@dataclass
class CaseRunResult:
    case: ResolvedCase
    status: str
    status_detail: str
    signals: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    actions: pd.DataFrame
    daily: pd.DataFrame
    holdings: pd.DataFrame
    calibration_rows: pd.DataFrame
    annual_returns: pd.DataFrame
    episodes: pd.DataFrame
    metrics: Dict[str, object]
    data_failures: List[Dict[str, object]]


def _week_signal_sessions(sessions: pd.DatetimeIndex) -> List[pd.Timestamp]:
    seen = set()
    out = []
    for session in sessions:
        key = (session.isocalendar().year, session.isocalendar().week)
        if key in seen:
            continue
        seen.add(key)
        out.append(pd.Timestamp(session))
    return out


def _bars_lookup(data: ResearchData, symbols: Sequence[str]) -> Dict[str, Dict[pd.Timestamp, Dict[str, float]]]:
    out: Dict[str, Dict[pd.Timestamp, Dict[str, float]]] = {}
    for symbol in symbols:
        frame = data.etf_bars(symbol)
        table: Dict[pd.Timestamp, Dict[str, float]] = {}
        for row in frame.itertuples(index=False):
            table[pd.Timestamp(row.trade_date)] = {
                "open": float(row.open), "high": float(row.high), "low": float(row.low),
                "close": float(row.close), "volume": float(row.volume_shares),
                "amount": float(row.amount_cny),
            }
        out[symbol] = table
    return out


def _price_with_carry(
    tables: Mapping[str, Mapping[pd.Timestamp, Mapping[str, float]]],
    symbol: str,
    session: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    max_carry: int,
) -> Tuple[Optional[float], Optional[pd.Timestamp], int]:
    """收盘价与已知最后价, 最多沿用 max_carry 个交易日。"""
    table = tables.get(symbol, {})
    if session in table and table[session]["close"] > 0:
        return table[session]["close"], session, 0
    position = sessions.get_loc(session)
    for back in range(1, max_carry + 1):
        idx = position - back
        if idx < 0:
            break
        prior = sessions[idx]
        if prior in table and table[prior]["close"] > 0:
            return table[prior]["close"], prior, back
    return None, None, max_carry + 1


def replay(
    data: ResearchData,
    protocol: Mapping[str, object],
    case: Mapping[str, object],
    actual_end: pd.Timestamp,
    start: Optional[pd.Timestamp] = None,
) -> CaseRunResult:
    """回放单个 case, 返回全部落盘用产物(不写文件)。"""
    resolved = resolve_case(protocol, case)
    initial = float(protocol["initial_capital"])
    exec_cfg = protocol["execution"]
    start = pd.Timestamp(start or protocol["research_start"])
    sessions = data.calendar_range(start, actual_end)
    if len(sessions) == 0:
        raise RuntimeError("回放区间没有交易日")
    start_session = sessions[0]
    signal_sessions = set(_week_signal_sessions(sessions))

    symbols = [resolved.symbol]
    if resolved.dividend_module:
        symbols.append(str(protocol["auxiliary_modules"]["B_pb_dividend"]["dividend_symbol"]))
    tables = _bars_lookup(data, symbols)

    portfolio = Portfolio(symbols, initial)
    state = StrategyState()
    dividend_state = StrategyState()
    pending: List[PendingOrder] = []
    signals_rows: List[Dict[str, object]] = []
    orders_rows: List[Dict[str, object]] = []
    trades_rows: List[Dict[str, object]] = []
    actions_rows: List[Dict[str, object]] = []
    daily_rows: List[Dict[str, object]] = []
    holdings_rows: List[Dict[str, object]] = []
    data_failures: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []
    open_episodes: Dict[str, Dict[str, object]] = {}
    quantity_history: Dict[str, Dict[pd.Timestamp, int]] = {s: {} for s in symbols}
    last_good_price: Dict[str, Tuple[Optional[float], Optional[pd.Timestamp], int]] = {s: (None, None, 99) for s in symbols}

    cal_params = calibration_params(protocol, resolved)
    book = CalibrationBook(
        data, cal_params, resolved.index_id, resolved.metric, resolved.strategy,
        q=resolved.exit_q, fixed_exit_level=resolved.fixed_exit_level,
    )
    snapshot: Optional[CalibrationSnapshot] = None
    snapshot_year: Optional[int] = None
    recalibrated = False
    calibration_block: Optional[Dict[str, object]] = None
    b_flat_session: Optional[pd.Timestamp] = None

    # 公司行动索引
    action_by_ex: Dict[pd.Timestamp, List[Dict[str, object]]] = {}
    for symbol in symbols:
        frame = data.etf_actions(symbol)
        for row in frame.itertuples(index=False):
            ex = pd.Timestamp(row.ex_date)
            action_by_ex.setdefault(ex, []).append({
                "event_id": row.event_id, "symbol": symbol, "type": row.type,
                "record_date": pd.Timestamp(row.record_date) if pd.notna(row.record_date) else None,
                "ex_date": ex,
                "payment_date": pd.Timestamp(row.payment_date) if pd.notna(row.payment_date) else None,
                "cash_per_share": float(row.cash_per_share) if pd.notna(row.cash_per_share) else 0.0,
                "share_factor": float(row.share_factor) if pd.notna(row.share_factor) else 1.0,
            })

    status = "historical_run_complete"
    status_detail = "ok"
    if getattr(data, "dataset_kind", "real") not in ("real",):
        # 合成数据只能验证实现流程, 不能记为真实历史完成
        status = "synthetic_pipeline_validated"
        status_detail = "synthetic_fixture_not_performance_evidence"
    order_counter = 0
    signal_positions = {s: data.session_pos(s) for s in sessions if s in signal_sessions}

    for session in sessions:
        portfolio.bought_today = {s: 0 for s in symbols}

        # 1) 公司行动(执行前先处理当日生效事件)
        for action in action_by_ex.get(session, []):
            symbol = action["symbol"]
            if action["type"] == "split":
                factor = float(action["share_factor"])
                if factor > 0 and abs(factor - 1.0) > 1e-12:
                    portfolio.quantity[symbol] = int(round(portfolio.quantity[symbol] * factor))
                    # cost_basis保存总取得成本；拆分改变单位成本，不改变总成本。
                actions_rows.append({
                    "event_id": action["event_id"], "symbol": symbol, "type": "split",
                    "ex_date": str(session.date()),
                    "record_date": str(action["record_date"].date()) if action["record_date"] is not None else "",
                    "payment_date": "",
                    "cash_per_share": 0.0, "share_factor": factor,
                    "quantity_entitled": portfolio.quantity[symbol], "amount": 0.0,
                    "recorded_session": str(session.date()), "paid_session": "",
                })
            elif action["type"] == "dividend" and action["cash_per_share"] > 0:
                record_date = action["record_date"] or sessions[max(0, data.session_pos(session) - 1)]
                qty = quantity_history.get(symbol, {}).get(pd.Timestamp(record_date), 0)
                amount = qty * float(action["cash_per_share"])
                if amount > 0:
                    # 周期损益按取得分红权利归属；到账时该周期可能已经卖清，
                    # 甚至同一标的已经开了新周期，不能把迟付分红记入新持仓。
                    owners = list(episode_rows) + list(open_episodes.values())
                    for episode in reversed(owners):
                        exit_date = episode.get("exit_session")
                        if (episode.get("symbol") == symbol and
                                pd.Timestamp(episode["entry_session"]) <= pd.Timestamp(record_date) and
                                (not exit_date or pd.Timestamp(exit_date) >= pd.Timestamp(record_date))):
                            episode["dividends"] = float(episode.get("dividends", 0.0) + amount)
                            if episode.get("status") == "closed":
                                episode["pnl"] = float(episode["pnl"] + amount)
                            break
                    portfolio.add_receivable(ReceivableBucket(
                        event_id=str(action["event_id"]), symbol=symbol, amount=amount,
                        payment_date=action["payment_date"],
                    ))
                actions_rows.append({
                    "event_id": action["event_id"], "symbol": symbol, "type": "dividend",
                    "ex_date": str(session.date()),
                    "record_date": str(pd.Timestamp(record_date).date()),
                    "payment_date": str(action["payment_date"].date()) if action["payment_date"] is not None else "",
                    "cash_per_share": float(action["cash_per_share"]), "share_factor": 1.0,
                    "quantity_entitled": int(qty), "amount": float(amount),
                    "recorded_session": str(session.date()), "paid_session": "",
                })

        settled = portfolio.settle_receivables(session)
        for item in settled:
            for row in actions_rows:
                if row["event_id"] == item["event_id"] and not row["paid_session"]:
                    row["paid_session"] = item["paid_session"]

        # 2) 执行窗口: 先卖后买
        todays = [o for o in pending if o.exec_session == session]
        pending = [o for o in pending if o.exec_session != session]
        todays.sort(key=lambda o: 0 if o.side == "sell" else 1)
        for order in todays:
            if any(action["symbol"] == order.symbol for action in action_by_ex.get(session, [])):
                order.status = "cancelled_corporate_action"
                orders_rows.append(order.as_row())
                continue
            bar = tables.get(order.symbol, {}).get(session)
            if bar is None or not np.isfinite(bar["open"]) or bar["open"] <= 0 or bar["volume"] <= 0:
                order.status = "unfilled_no_price_or_suspension"
                orders_rows.append(order.as_row())
                if order.side == "sell" and order.reason not in ("risk_equity_cap", "risk_cash_floor"):
                    if order.symbol == resolved.symbol:
                        state.phase = "exiting"
                    elif resolved.dividend_module and order.symbol == symbols[-1]:
                        dividend_state.phase = "exiting"
                continue
            raw_open = float(bar["open"])
            adv20 = order.adv20
            if order.side == "sell":
                available = portfolio.sellable_quantity(order.symbol)
                if available <= 0:
                    order.status = "unfilled_no_sellable_quantity"
                    orders_rows.append(order.as_row())
                    continue
                want = available if order.requested_qty is None else min(int(order.requested_qty), available)
                cap_shares = int((exec_cfg["max_participation"] * adv20) / raw_open) if adv20 else want
                fill_qty = min(want, max(0, cap_shares)) if adv20 else want
                if fill_qty <= 0:
                    order.status = "unfilled_participation_cap"
                    orders_rows.append(order.as_row())
                    continue
                notional_raw = fill_qty * raw_open
                impact = _impact(notional_raw, adv20, exec_cfg["impact_coef"])
                price = raw_open * (1 - resolved.slippage - impact)
                notional = fill_qty * price
                fee = _commission(notional, exec_cfg["commission_rate"], exec_cfg["min_commission"])
                proceeds = notional - fee
                portfolio.cash += proceeds
                portfolio.total_fees += fee
                portfolio.total_turnover += notional
                portfolio.quantity[order.symbol] -= fill_qty
                cost_removed = portfolio.cost_basis[order.symbol] * (fill_qty / max(1, fill_qty + portfolio.quantity[order.symbol])) \
                    if portfolio.quantity[order.symbol] >= 0 else 0.0
                remaining = portfolio.quantity[order.symbol]
                if remaining == 0:
                    cost_removed = portfolio.cost_basis[order.symbol]
                    portfolio.cost_basis[order.symbol] = 0.0
                else:
                    portfolio.cost_basis[order.symbol] = max(0.0, portfolio.cost_basis[order.symbol] - cost_removed)
                portfolio.realized_pnl[order.symbol] = portfolio.realized_pnl.get(order.symbol, 0.0) + proceeds - cost_removed
                order.status = "filled" if fill_qty == want else "partial"
                order.filled_qty = fill_qty
                order.fill_price = price
                order.commission = fee
                order.impact = impact
                orders_rows.append(order.as_row())
                trades_rows.append({
                    "trade_id": f"{order.order_id}-T", "order_id": order.order_id,
                    "signal_date": str(pd.Timestamp(order.signal_session).date()),
                    "execution_date": str(session.date()), "side": "sell", "symbol": order.symbol,
                    "reason": order.reason, "quantity": fill_qty, "price": price,
                    "notional": notional, "commission": fee, "impact": impact,
                    "slippage": resolved.slippage, "adv20": adv20,
                    "cash_after": portfolio.cash, "quantity_after": portfolio.quantity[order.symbol],
                    "status": order.status,
                })
                if remaining == 0:
                    episode = open_episodes.pop(order.symbol, None)
                    if episode is not None:
                        episode["exit_session"] = str(session.date())
                        episode["exit_notional"] = float(episode.get("exit_notional", 0.0) + notional)
                        episode["exit_fees"] = float(episode.get("exit_fees", 0.0) + fee)
                        episode["status"] = "closed"
                        episode["pnl"] = float(episode["sell_notional"] - episode["buy_notional"]
                                              + episode.get("dividends", 0.0) - episode.get("fees", 0.0))
                        episode["holding_days"] = int((session - pd.Timestamp(episode["entry_session"])).days)
                        episode_rows.append(episode)
                    if order.symbol == resolved.symbol:
                        # 风险修正可以清空份额, 但不能清零本轮分批计数与冷却。
                        if order.reason not in ("risk_equity_cap", "risk_cash_floor"):
                            state.reset_round()
                        else:
                            state.phase = "holding" if state.buy_steps > 0 else "flat"
                        b_flat_session = session
                    elif resolved.dividend_module and order.symbol == symbols[-1]:
                        dividend_state.reset_round()
            else:  # buy
                if order.requires_symbol_flat:
                    if portfolio.quantity.get(order.requires_symbol_flat, 0) > 0:
                        order.status = "cancelled_dependency_not_flat"
                        orders_rows.append(order.as_row())
                        continue
                if order.adv20 is None or order.adv20 <= 0:
                    order.status = "unfilled_no_adv20"
                    orders_rows.append(order.as_row())
                    continue
                prices_open = {s: (tables.get(s, {}).get(session, {}).get("open", 0.0) or 0.0) for s in symbols}
                nav_open = portfolio.cash + sum(portfolio.quantity[s] * prices_open[s] for s in symbols) + portfolio.receivable
                equity_open = sum(portfolio.quantity[s] * prices_open[s] for s in symbols)
                want = int(order.requested_qty or 0)
                if want <= 0:
                    order.status = "cancelled_zero_request"
                    orders_rows.append(order.as_row())
                    continue
                impact_guess = _impact(want * raw_open, order.adv20, exec_cfg["impact_coef"])
                price = raw_open * (1 + resolved.slippage + impact_guess)
                max_by_cash = _max_buy_lots_with_cash_floor(
                    portfolio.cash,
                    nav_open,
                    price,
                    float(exec_cfg["min_cash_share"]),
                    int(exec_cfg["lot_size"]),
                    float(exec_cfg["commission_rate"]),
                    float(exec_cfg["min_commission"]),
                )
                cap_shares = int((exec_cfg["max_participation"] * order.adv20) / raw_open)
                room = max(0.0, resolved.w_cap * nav_open - equity_open)
                max_by_cap = int(room / price)
                fill_qty = min(want, max_by_cash, cap_shares, max_by_cap)
                fill_qty = (fill_qty // exec_cfg["lot_size"]) * exec_cfg["lot_size"]
                if fill_qty <= 0:
                    order.status = "unfilled_constraints"
                    orders_rows.append(order.as_row())
                    continue
                notional_raw = fill_qty * raw_open
                impact = _impact(notional_raw, order.adv20, exec_cfg["impact_coef"])
                price = raw_open * (1 + resolved.slippage + impact)
                notional = fill_qty * price
                fee = _commission(notional, exec_cfg["commission_rate"], exec_cfg["min_commission"])
                total_cost = notional + fee
                if total_cost > portfolio.cash:
                    fill_qty -= exec_cfg["lot_size"]
                    if fill_qty <= 0:
                        order.status = "unfilled_cash"
                        orders_rows.append(order.as_row())
                        continue
                    notional = fill_qty * price
                    fee = _commission(notional, exec_cfg["commission_rate"], exec_cfg["min_commission"])
                    total_cost = notional + fee
                portfolio.cash -= total_cost
                portfolio.total_fees += fee
                portfolio.total_turnover += notional
                portfolio.quantity[order.symbol] = portfolio.quantity.get(order.symbol, 0) + fill_qty
                portfolio.bought_today[order.symbol] = portfolio.bought_today.get(order.symbol, 0) + fill_qty
                portfolio.cost_basis[order.symbol] = portfolio.cost_basis.get(order.symbol, 0.0) + total_cost
                order.status = "filled" if fill_qty == want else "partial"
                order.filled_qty = fill_qty
                order.fill_price = price
                order.commission = fee
                order.impact = impact
                orders_rows.append(order.as_row())
                trades_rows.append({
                    "trade_id": f"{order.order_id}-T", "order_id": order.order_id,
                    "signal_date": str(pd.Timestamp(order.signal_session).date()),
                    "execution_date": str(session.date()), "side": "buy", "symbol": order.symbol,
                    "reason": order.reason, "quantity": fill_qty, "price": price,
                    "notional": notional, "commission": fee, "impact": impact,
                    "slippage": resolved.slippage, "adv20": order.adv20,
                    "cash_after": portfolio.cash, "quantity_after": portfolio.quantity[order.symbol],
                    "status": order.status,
                })
                if fill_qty > 0:
                    episode = open_episodes.get(order.symbol)
                    if episode is None:
                        episode = {
                            "symbol": order.symbol, "entry_session": str(session.date()),
                            "buy_notional": 0.0, "sell_notional": 0.0, "fees": 0.0,
                            "dividends": 0.0, "status": "open", "pnl": None,
                        }
                        open_episodes[order.symbol] = episode
                    episode["buy_notional"] = float(episode["buy_notional"] + notional)
                    episode["fees"] = float(episode["fees"] + fee)
                    # 只有实际成交才推进分批计数; 部分成交同样只算一批
                    if order.symbol == resolved.symbol and order.reason.startswith("buy_step"):
                        state.buy_steps = min(3, state.buy_steps + 1)
                        state.last_buy_session = session
                        state.phase = "holding"
                    elif resolved.dividend_module and order.symbol == symbols[-1]:
                        dividend_state.phase = "holding"

        # 3) 收盘记账
        prices: Dict[str, float] = {}
        stale_flags: List[str] = []
        for symbol in symbols:
            price, price_date, carried = _price_with_carry(
                tables, symbol, session, sessions, int(exec_cfg["max_price_carry_sessions"])
            )
            if price is None:
                # 只有持仓需要估值; 空仓标的不产生 unpriceable_holding
                if portfolio.quantity.get(symbol, 0) > 0:
                    data_failures.append({
                        "session": str(session.date()), "symbol": symbol,
                        "code": "unpriceable_holding", "carried_sessions": carried,
                        "quantity": portfolio.quantity[symbol],
                    })
                    status = "data_or_scope_blocked"
                    status_detail = f"unpriceable_holding:{symbol}:{session.date()}"
                price = 0.0
            else:
                if carried > 0:
                    stale_flags.append(f"{symbol}:carry{carried}")
            prices[symbol] = float(price)
            last_good_price[symbol] = (price, price_date, carried)

        for symbol in symbols:
            quantity_history[symbol][session] = int(portfolio.quantity[symbol])

        equity_value = portfolio.market_value(prices)
        nav = portfolio.nav(prices)
        daily_rows.append({
            "session": str(session.date()),
            "cash": round(portfolio.cash, 6),
            "dividend_receivable": round(portfolio.receivable, 6),
            **{f"quantity_{s}": int(portfolio.quantity.get(s, 0)) for s in symbols},
            **{f"price_{s}": float(prices.get(s, 0.0)) for s in symbols},
            "equity_value": round(equity_value, 6),
            "nav": round(nav, 6),
            "equity_weight": round(equity_value / nav, 8) if nav > 0 else 0.0,
            "cash_share": round(portfolio.cash / nav, 8) if nav > 0 else 0.0,
            "cost_basis": round(sum(portfolio.cost_basis.values()), 6),
            "cumulative_fees": round(portfolio.total_fees, 6),
            "cumulative_turnover": round(portfolio.total_turnover, 6),
            "stale_prices": ";".join(stale_flags),
            "missing_data": 0 if not stale_flags else 1,
        })
        holdings_rows.append({
            "session": str(session.date()),
            **{f"{s}_qty": int(portfolio.quantity.get(s, 0)) for s in symbols},
            **{f"{s}_value": round(portfolio.value(s, prices), 2) for s in symbols},
            "receivable": round(portfolio.receivable, 6),
        })
        if status not in ("historical_run_complete", "synthetic_pipeline_validated"):
            break

        # 4) 周度信号(19:00, 收盘后)
        if session in signal_sessions:
            year = int(session.year)
            snapshot = book.get(year, snapshot_ts=session_ts(session))
            if snapshot is None:
                calibration_block = book.failures.get(year)
            else:
                if snapshot_year is not None and snapshot.year != snapshot_year:
                    recalibrated = True
                snapshot_year = snapshot.year

            def build_account() -> AccountSnapshot:
                quantities = {s: int(portfolio.quantity.get(s, 0)) for s in symbols}
                valuation_prices: Dict[str, float] = {}
                trade_prices: Dict[str, float] = {}
                for item in symbols:
                    carried_price, _, _ = _price_with_carry(
                        tables,
                        item,
                        session,
                        sessions,
                        int(exec_cfg["max_price_carry_sessions"]),
                    )
                    valuation_prices[item] = float(carried_price or 0.0)
                    raw_close = tables.get(item, {}).get(session, {}).get("close", 0.0) or 0.0
                    trade_prices[item] = float(raw_close) if np.isfinite(raw_close) and raw_close > 0 else 0.0
                nav_signal = (
                    portfolio.cash
                    + sum(quantities[item] * valuation_prices[item] for item in symbols)
                    + portfolio.receivable
                )
                return AccountSnapshot(
                    nav=nav_signal, cash=portfolio.cash, dividend_receivable=portfolio.receivable,
                    quantities=quantities, prices=valuation_prices, trade_prices=trade_prices,
                )

            account_snapshot = build_account()
            valuation = data.valuation_asof(resolved.index_id, resolved.metric, session)
            gate = None
            if resolved.gate == "ratio":
                gate = ratio_gate_series(
                    data, session,
                    protocol["auxiliary_modules"]["A_pe_ratio_gate"]["ratio_base_date"],
                    float(protocol["auxiliary_modules"]["A_pe_ratio_gate"]["ratio_threshold"]),
                )
            dividend_yield = None
            if resolved.dividend_module:
                dividend_yield = data.yield_asof(
                    str(protocol["auxiliary_modules"]["B_pb_dividend"]["dividend_index"]), session
                )
            ctx = SignalContext(
                strategy=resolved.strategy,
                session=session,
                data=data,
                snapshot=snapshot,
                calibration_failure=calibration_block,
                valuation=valuation,
                account=account_snapshot,
                state=state,
                primary_symbol=resolved.symbol,
                w_cap=resolved.w_cap,
                cooldown_sessions=resolved.cooldown_sessions,
                exit_q=resolved.exit_q,
                max_valuation_lag_sessions=int(exec_cfg["max_valuation_lag_sessions"]),
                min_cash_share=float(exec_cfg["min_cash_share"]),
                equity_cap_tolerance=float(exec_cfg["equity_cap_tolerance"]),
                buy_budget_nav_share=float(exec_cfg["buy_budget_nav_share"]),
                commission_rate=float(exec_cfg["commission_rate"]),
                min_commission=float(exec_cfg["min_commission"]),
                slippage=resolved.slippage,
                recalibrated=recalibrated,
                ratio_gate=gate,
                gate_required=(resolved.gate == "ratio"),
                dividend_yield=dividend_yield,
                dividend_symbol=(symbols[1] if len(symbols) > 1 else None),
            )
            decision = evaluate_weekly(ctx)
            recalibrated = False

            next_pos = data.session_pos(session) + resolved.delay_sessions
            exec_session = data.calendar[next_pos] if next_pos < len(data.calendar) and data.calendar[next_pos] <= actual_end else None
            planned: List[OrderIntent] = list(decision.orders)

            # 红利模块: 创业板优先, 未卖清不得买创业板
            if resolved.dividend_module:
                div_symbol = symbols[1]
                b_symbol = resolved.symbol
                div_held = portfolio.quantity.get(div_symbol, 0)
                b_held = portfolio.quantity.get(b_symbol, 0)
                b_buy = any(o.side == "buy" and o.symbol == b_symbol for o in planned)
                b_valuation_ok = (
                    bool(valuation)
                    and valuation.get("lag_sessions", 99) <= int(exec_cfg["max_valuation_lag_sessions"])
                    and np.isfinite(float(valuation.get("value", -1)))
                    and float(valuation.get("value", -1)) > 0
                )
                flat_allowed = (b_held == 0 and state.phase != "exiting"
                                and b_valuation_ok and session != b_flat_session)
                if b_buy and div_held > 0:
                    planned = [OrderIntent(side="sell", symbol=div_symbol, reason="dividend_switch_sell",
                                           requested_qty=None, note="为创业板买入腾挪, 先卖清红利")]
                    dividend_state.phase = "exiting"
                    dividend_state.exit_reason = "dividend_switch_sell"
                    for order in decision.orders:
                        if order.side == "buy" and order.symbol == b_symbol:
                            order.requires_symbol_flat = div_symbol
                            planned.append(order)
                    decision.checks["dividend_reason"] = "dividend_switch_sell"
                    decision.detail = f"{decision.detail} | 红利: 先卖清再尝试创业板买入"
                elif b_held > 0 and div_held > 0 and state.phase != "exiting":
                    planned.append(OrderIntent(side="sell", symbol=div_symbol,
                                               reason="dividend_priority_exit", requested_qty=None,
                                               note="创业板持仓存在, 红利不新增"))
                    dividend_state.phase = "exiting"
                    dividend_state.exit_reason = "dividend_priority_exit"
                    if not decision.orders:
                        decision.reason = "dividend_priority_exit"
                        decision.detail = "创业板持仓优先, 红利退出"
                    else:
                        decision.detail = f"{decision.detail} | 红利: 创业板持仓优先退出"
                elif div_held > 0 or flat_allowed:
                    div_ctx = SignalContext(**{
                        **ctx.__dict__,
                        "strategy": "B_pb_dividend",
                        "dividend_state": dividend_state,
                    })
                    div_ctx.dividend_gate_open = flat_allowed
                    div_decision = evaluate_dividend_module(div_ctx)
                    if div_decision.orders:
                        planned.extend(div_decision.orders)
                        decision.checks["dividend_reason"] = div_decision.reason
                        if not decision.orders:
                            decision.reason = div_decision.reason
                            decision.detail = div_decision.detail
                        else:
                            decision.detail = f"{decision.detail} | 红利: {div_decision.detail}"
            decision.orders = list(planned)

            signals_rows.append({**decision.as_row(), "exec_session": str(exec_session.date()) if exec_session else "unexecuted_at_end"})
            for intent in planned:
                if exec_session is None:
                    order_counter += 1
                    stale = PendingOrder(
                        order_id=f"O{order_counter:05d}", signal_session=session,
                        exec_session=session, side=intent.side, symbol=intent.symbol,
                        reason=intent.reason, requested_qty=intent.requested_qty,
                        reference_price=float(account_snapshot.prices.get(intent.symbol, 0.0)),
                        adv20=data.adv20(intent.symbol, session),
                        target_weight=intent.target_weight,
                        requires_symbol_flat=intent.requires_symbol_flat,
                        status="unexecuted_at_end", note=intent.note,
                    )
                    orders_rows.append(stale.as_row())
                    continue
                order_counter += 1
                pending.append(PendingOrder(
                    order_id=f"O{order_counter:05d}", signal_session=session, exec_session=exec_session,
                    side=intent.side, symbol=intent.symbol, reason=intent.reason,
                    requested_qty=intent.requested_qty,
                    reference_price=float(account_snapshot.prices.get(intent.symbol, 0.0)),
                    adv20=data.adv20(intent.symbol, session),
                    target_weight=intent.target_weight,
                    requires_symbol_flat=intent.requires_symbol_flat,
                    note=intent.note,
                ))

    # 期末仍在途订单
    for order in pending:
        order.status = "unexecuted_at_end"
        orders_rows.append(order.as_row())

    # 未平仓周期
    for symbol, episode in open_episodes.items():
        episode = dict(episode)
        episode["exit_session"] = ""
        episode["status"] = "open"
        episode["pnl"] = None
        episode["holding_days"] = int((sessions[-1] - pd.Timestamp(episode["entry_session"])).days)
        episode_rows.append(episode)

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        daily = pd.DataFrame(columns=["session", "nav", "cash"])
    daily["session"] = pd.to_datetime(daily["session"])
    metrics = compute_metrics(
        daily, pd.DataFrame(trades_rows), pd.DataFrame(episode_rows), initial,
        sessions[0], sessions[-1], resolved, protocol,
    )
    if calibration_block is not None and status in ("historical_run_complete", "synthetic_pipeline_validated"):
        status = "data_or_scope_blocked"
        status_detail = f"calibration_data_blocked:{calibration_block.get('reason', 'unknown')}"

    return CaseRunResult(
        case=resolved,
        status=status,
        status_detail=status_detail,
        signals=pd.DataFrame(signals_rows),
        orders=pd.DataFrame(orders_rows),
        trades=pd.DataFrame(trades_rows),
        actions=pd.DataFrame(actions_rows),
        daily=daily,
        holdings=pd.DataFrame(holdings_rows),
        calibration_rows=pd.DataFrame(book.rows()),
        annual_returns=annual_returns(daily, initial),
        episodes=pd.DataFrame(episode_rows),
        metrics=metrics,
        data_failures=data_failures,
    )


# ------------------------------------------------------------------ 基准
def run_benchmark(
    data: ResearchData,
    protocol: Mapping[str, object],
    bench: Mapping[str, object],
    actual_end: pd.Timestamp,
    start: pd.Timestamp,
    data_gaps: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """静态基准: 纯现金或首个可执行开盘一次性建仓后持有。

    仍遵守真实现金、费用、手数、ADV 参与率与部分成交剩余取消, 不做周度恢复权重。
    """
    initial = float(protocol["initial_capital"])
    exec_cfg = protocol["execution"]
    sessions = data.calendar_range(start, actual_end)
    kind = str(bench["kind"])
    bench_id = str(bench["id"])
    if kind == "cash":
        daily = pd.DataFrame({
            "session": sessions,
            "cash": initial, "dividend_receivable": 0.0,
            "equity_value": 0.0, "nav": initial, "equity_weight": 0.0, "cash_share": 1.0,
            "cost_basis": 0.0, "cumulative_fees": 0.0, "cumulative_turnover": 0.0,
            "stale_prices": "", "missing_data": 0,
        })
        metrics = compute_metrics(
            daily, pd.DataFrame(), pd.DataFrame(), initial, sessions[0], sessions[-1],
            ResolvedCase(bench_id, "benchmark", "A", "benchmark", "", "", "", 0.0, 0.0, 1.1,
                         20, 1, 0.0005, None, None, False, {}),
            protocol,
        )
        metrics["benchmark"] = {"id": bench_id, "kind": kind, "initial_weight": 0.0,
                                "actual_initial_position": 0.0}
        return {"daily": daily, "trades": pd.DataFrame(), "actions": pd.DataFrame(), "metrics": metrics}

    strategy = str(bench["strategy"])
    base = dict(protocol["strategies"][strategy])
    symbol = str(base["etf"])
    initial_weight = float(bench["initial_weight"])
    tables = _bars_lookup(data, [symbol])
    portfolio = Portfolio([symbol], initial)
    rows: List[Dict[str, object]] = []
    trades: List[Dict[str, object]] = []
    action_by_ex: Dict[pd.Timestamp, List[Dict[str, object]]] = {}
    for row in data.etf_actions(symbol).itertuples(index=False):
        action_by_ex.setdefault(pd.Timestamp(row.ex_date), []).append({
            "event_id": row.event_id, "type": row.type,
            "record_date": pd.Timestamp(row.record_date) if pd.notna(row.record_date) else None,
            "payment_date": pd.Timestamp(row.payment_date) if pd.notna(row.payment_date) else None,
            "cash_per_share": float(row.cash_per_share) if pd.notna(row.cash_per_share) else 0.0,
            "share_factor": float(row.share_factor) if pd.notna(row.share_factor) else 1.0,
        })
    quantity_history: Dict[pd.Timestamp, int] = {}
    actions_rows: List[Dict[str, object]] = []
    entered = False
    first_session = sessions[0]
    for session in sessions:
        for action in action_by_ex.get(session, []):
            if action["type"] == "split" and action["share_factor"] > 0:
                portfolio.quantity[symbol] = int(round(portfolio.quantity[symbol] * action["share_factor"]))
            elif action["type"] == "dividend" and action["cash_per_share"] > 0:
                record = action["record_date"] or session
                qty = quantity_history.get(pd.Timestamp(record), 0)
                amount = qty * float(action["cash_per_share"])
                if qty > 0:
                    portfolio.add_receivable(ReceivableBucket(
                        event_id=str(action["event_id"]), symbol=symbol,
                        amount=amount, payment_date=action["payment_date"],
                    ))
                actions_rows.append({
                    "event_id": action["event_id"], "symbol": symbol, "type": "dividend",
                    "ex_date": str(session.date()),
                    "record_date": str(pd.Timestamp(record).date()),
                    "payment_date": str(action["payment_date"].date()) if action["payment_date"] is not None else "",
                    "cash_per_share": float(action["cash_per_share"]), "share_factor": 1.0,
                    "quantity_entitled": int(qty), "amount": float(amount),
                    "recorded_session": str(session.date()), "paid_session": "",
                })
        for item in portfolio.settle_receivables(session):
            for row in actions_rows:
                if row["event_id"] == item["event_id"] and not row["paid_session"]:
                    row["paid_session"] = item["paid_session"]
        bar = tables.get(symbol, {}).get(session)
        if not entered and session > first_session and bar is not None and bar["open"] > 0 and bar["volume"] > 0:
            adv20 = data.adv20(symbol, session)
            if adv20:
                prices_open = {symbol: float(bar["open"])}
                nav_open = portfolio.cash + portfolio.market_value(prices_open) + portfolio.receivable
                budget = initial_weight * nav_open
                impact = _impact(budget, adv20, exec_cfg["impact_coef"])
                price = bar["open"] * (1 + exec_cfg["slippage"] + impact)
                cap_shares = int((exec_cfg["max_participation"] * adv20) / bar["open"])
                lots = int(min(budget, portfolio.cash) / price // exec_cfg["lot_size"]) * exec_cfg["lot_size"]
                fill = min(lots, cap_shares)
                fill = (fill // exec_cfg["lot_size"]) * exec_cfg["lot_size"]
                if fill > 0:
                    notional = fill * price
                    fee = _commission(notional, exec_cfg["commission_rate"], exec_cfg["min_commission"])
                    portfolio.cash -= notional + fee
                    portfolio.total_fees += fee
                    portfolio.quantity[symbol] += fill
                    portfolio.cost_basis[symbol] = notional + fee
                    trades.append({
                        "trade_id": "B001-T", "order_id": "B001",
                        "signal_date": str(session.date()), "execution_date": str(session.date()),
                        "side": "buy", "symbol": symbol, "reason": "benchmark_initial",
                        "quantity": fill, "price": price, "notional": notional, "commission": fee,
                        "impact": impact, "slippage": exec_cfg["slippage"], "adv20": adv20,
                        "cash_after": portfolio.cash, "quantity_after": portfolio.quantity[symbol],
                        "status": "filled" if fill == lots else "partial",
                    })
                entered = True
        quantity_history[session] = int(portfolio.quantity[symbol])
        price, _, price_lag = _price_with_carry(
            tables, symbol, session, sessions, int(exec_cfg["max_price_carry_sessions"]))
        if price is None and portfolio.quantity[symbol] > 0:
            raise ValueError(f"unpriceable_benchmark_holding:{symbol}:{session.date()}")
        price = float(price or 0.0)
        prices = {symbol: price}
        nav = portfolio.nav(prices)
        rows.append({
            "session": session, "cash": portfolio.cash, "dividend_receivable": portfolio.receivable,
            f"quantity_{symbol}": int(portfolio.quantity[symbol]), f"price_{symbol}": float(price),
            "equity_value": portfolio.market_value(prices), "nav": nav,
            "equity_weight": portfolio.market_value(prices) / nav if nav > 0 else 0.0,
            "cash_share": portfolio.cash / nav if nav > 0 else 0.0,
            "cost_basis": float(portfolio.cost_basis[symbol]), "cumulative_fees": portfolio.total_fees,
            "cumulative_turnover": portfolio.total_turnover,
            "stale_prices": symbol if price_lag > 0 and portfolio.quantity[symbol] > 0 else "",
            "missing_data": int(price_lag > 0 and portfolio.quantity[symbol] > 0),
        })
    daily = pd.DataFrame(rows)
    case = ResolvedCase(bench_id, "benchmark", strategy, "benchmark", str(base["index_id"]), str(base["metric"]),
                        symbol, float(base["w_cap"]), float(base["exit_q"]), 1.1, 20, 1,
                        float(exec_cfg["slippage"]), None, None, False, dict(base))
    metrics = compute_metrics(daily, pd.DataFrame(trades), pd.DataFrame(), initial, sessions[0], sessions[-1], case, protocol)
    entry_weight = 0.0
    column = f"quantity_{symbol}"
    if not daily.empty and column in daily:
        held = daily[daily[column].astype(float) > 0]
        if not held.empty:
            entry_weight = float(held["equity_weight"].iloc[0])
    metrics["benchmark"] = {"id": bench_id, "kind": kind, "initial_weight": initial_weight,
                            "actual_initial_position": entry_weight}
    return {"daily": daily, "trades": pd.DataFrame(trades), "actions": pd.DataFrame(actions_rows),
            "metrics": metrics}


# ------------------------------------------------------------------ 指标
def annual_returns(daily: pd.DataFrame, initial: float) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["year", "return", "full_year", "start_nav", "end_nav", "sessions"])
    frame = daily.copy()
    frame["year"] = frame["session"].dt.year
    rows: List[Dict[str, object]] = []
    previous_end = initial
    years = sorted(frame["year"].unique())
    for year in years:
        block = frame[frame["year"] == year]
        end_nav = float(block["nav"].iloc[-1])
        ret = end_nav / previous_end - 1 if previous_end > 0 else float("nan")
        first = block["session"].iloc[0]
        last = block["session"].iloc[-1]
        full = bool(first.month == 1 and first.day <= 5 and last.month == 12)
        rows.append({
            "year": int(year),
            "return": ret,
            "full_year": full,
            "start_nav": previous_end,
            "end_nav": end_nav,
            "sessions": int(len(block)),
            "start_session": str(first.date()),
            "end_session": str(last.date()),
        })
        previous_end = end_nav
    return pd.DataFrame(rows)


def compute_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    episodes: pd.DataFrame,
    initial: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    case: ResolvedCase,
    protocol: Mapping[str, object],
) -> Dict[str, object]:
    if daily.empty or "nav" not in daily:
        return {
            "case_id": case.case_id, "status": "no_data", "final_nav": initial,
            "cagr": None, "max_drawdown": None, "worst_full_year": None,
        }
    nav = daily["nav"].astype(float)
    final_nav = float(nav.iloc[-1])
    days = max(1, int((pd.Timestamp(end) - pd.Timestamp(start)).days))
    cagr = (final_nav / initial) ** (365.2425 / days) - 1 if final_nav > 0 else float("nan")
    # 初始资金计入高水位: 序列首元素为 initial, 之后为每日净值
    series = np.concatenate([[float(initial)], nav.to_numpy()])
    high_water = np.maximum.accumulate(series)
    dd = series / high_water - 1
    trough_pos = int(np.argmin(dd))
    max_dd = float(max(0.0, -float(dd.min())))
    peak_pos = int(np.argmax(series[: trough_pos + 1]))
    peak_date = pd.Timestamp(start) if peak_pos == 0 else pd.Timestamp(daily["session"].iloc[peak_pos - 1])
    peak_value = float(series[peak_pos])
    trough_date = pd.Timestamp(daily["session"].iloc[max(0, trough_pos - 1)])
    recovery_date = None
    for idx in range(trough_pos + 1, len(series)):
        if series[idx] >= peak_value:
            recovery_date = pd.Timestamp(daily["session"].iloc[idx - 1])
            break
    annual = annual_returns(daily, initial)
    full = annual[annual["full_year"]] if not annual.empty else annual
    worst_full = float(full["return"].min()) if not full.empty else None
    worst_full_year = int(full.loc[full["return"].idxmin(), "year"]) if not full.empty else None
    rolling = nav.pct_change(252).dropna()
    worst_rolling = float(rolling.min()) if len(rolling) else None
    equity_weight = daily["equity_weight"].astype(float)
    cash_share = daily["cash_share"].astype(float)
    closed = episodes[episodes.get("status") == "closed"] if not episodes.empty and "status" in episodes else pd.DataFrame()
    open_count = int((episodes["status"] == "open").sum()) if not episodes.empty and "status" in episodes else 0
    n_trades = int(len(trades)) if not trades.empty else 0
    overlimit_days = int((equity_weight > case.w_cap + 1e-9).sum())
    goals = {
        "cagr": cagr,
        "worst_full_year": worst_full,
        "max_drawdown": max_dd,
    }
    cagr_pass = bool(cagr is not None and cagr >= 0.135)
    annual_loss_pass = bool(worst_full is not None and worst_full >= -0.05)
    drawdown_pass = bool(max_dd <= 0.40)
    return {
        "case_id": case.case_id,
        "strategy": case.strategy,
        "group": case.group,
        "window": {"start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()), "days": days},
        "initial_capital": initial,
        "final_nav": final_nav,
        "cumulative_return": final_nav / initial - 1,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "max_drawdown_period": {"peak": str(peak_date.date()), "trough": str(trough_date.date()),
                                "recovery": str(recovery_date.date()) if recovery_date is not None else None},
        "worst_full_year": worst_full,
        "worst_full_year_year": worst_full_year,
        "annual_returns": annual.to_dict(orient="records"),
        "worst_rolling_12m": worst_rolling,
        "avg_equity_weight": float(equity_weight.mean()),
        "max_equity_weight": float(equity_weight.max()),
        "avg_cash_share": float(cash_share.mean()),
        "days_above_cap": overlimit_days,
        "trade_count": n_trades,
        "closed_episodes": int(len(closed)),
        "open_episodes": open_count,
        "avg_closed_holding_days": float(closed["holding_days"].mean()) if not closed.empty else None,
        "total_fees": float(trades["commission"].sum()) if n_trades else 0.0,
        "total_turnover": float(trades["notional"].sum()) if n_trades else 0.0,
        "goals": {
            "cagr_pass": cagr_pass,
            "annual_loss_pass": annual_loss_pass,
            "max_drawdown_pass": drawdown_pass,
            "joint_pass": bool(cagr_pass and annual_loss_pass and drawdown_pass),
            "note": "仅本口径历史达标, 不等于投资准入",
        },
    }
