"""周度决策状态机: 退出优先、风险限额、低估分批买入、股债门槛与红利切换。

严格按 README 第5-7节的优先级:
已有退出意图 → 资格退出 → 高估清仓 → 数据有效性 → 风险限额修正 → 低估分批买入 → 持有不动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .calibration import CalibrationSnapshot
from .data import ResearchData, session_ts


# ------------------------------------------------------------------ 数据结构
@dataclass
class OrderIntent:
    """周度信号产生的订单意图(尚未成交)。"""

    side: str                      # buy | sell
    symbol: str
    reason: str
    requested_qty: Optional[int]   # None 表示按全部可卖份额清仓
    target_weight: Optional[float] = None
    requires_symbol_flat: Optional[str] = None
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "side": self.side,
            "symbol": self.symbol,
            "reason": self.reason,
            "requested_qty": self.requested_qty,
            "target_weight": self.target_weight,
            "requires_symbol_flat": self.requires_symbol_flat,
            "note": self.note,
        }


@dataclass
class AccountSnapshot:
    """信号时点(收盘后)的账户只读快照。"""

    nav: float
    cash: float
    dividend_receivable: float
    quantities: Dict[str, int]
    prices: Dict[str, float]
    trade_prices: Optional[Dict[str, float]] = None

    def value(self, symbol: str) -> float:
        return float(self.quantities.get(symbol, 0)) * float(self.prices.get(symbol, 0.0))

    def trade_price(self, symbol: str) -> float:
        prices = self.trade_prices if self.trade_prices is not None else self.prices
        return float(prices.get(symbol, 0.0))

    @property
    def equity_value(self) -> float:
        return float(sum(self.value(s) for s in self.quantities))

    @property
    def equity_weight(self) -> float:
        return self.equity_value / self.nav if self.nav > 0 else 0.0

    @property
    def cash_share(self) -> float:
        return self.cash / self.nav if self.nav > 0 else 0.0


@dataclass
class StrategyState:
    phase: str = "flat"                 # flat | holding | exiting
    buy_steps: int = 0
    last_buy_session: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None

    def reset_round(self):
        self.phase = "flat"
        self.buy_steps = 0
        self.last_buy_session = None
        self.exit_reason = None


@dataclass
class SignalContext:
    strategy: str
    session: pd.Timestamp
    data: ResearchData
    snapshot: Optional[CalibrationSnapshot]
    calibration_failure: Optional[Mapping[str, object]]
    valuation: Optional[Mapping[str, object]]
    account: AccountSnapshot
    state: StrategyState
    primary_symbol: str
    w_cap: float
    cooldown_sessions: int
    exit_q: float
    max_valuation_lag_sessions: int = 1
    min_cash_share: float = 0.05
    equity_cap_tolerance: float = 0.03
    buy_budget_nav_share: float = 0.30
    commission_rate: float = 0.0003
    min_commission: float = 0.20
    slippage: float = 0.0005
    recalibrated: bool = False
    ratio_gate: Optional[Mapping[str, object]] = None
    gate_required: bool = False
    dividend_yield: Optional[Mapping[str, object]] = None
    dividend_symbol: Optional[str] = None
    dividend_gate_open: bool = False
    dividend_state: Optional[StrategyState] = None
    extra_symbols: Sequence[str] = ()
    note: str = ""


@dataclass
class Decision:
    strategy: str
    session: str
    timestamp: str
    phase_before: str
    phase_after_intent: str
    reason: str
    detail: str
    value: Optional[float]
    valuation_date: Optional[str]
    valuation_available_at: Optional[str]
    valuation_lag: Optional[int]
    thresholds: Dict[str, float]
    allowed_steps: int
    buy_steps_before: int
    buy_steps_after_intent: int
    orders: List[OrderIntent] = field(default_factory=list)
    skips: List[str] = field(default_factory=list)
    checks: Dict[str, object] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "session": self.session,
            "timestamp": self.timestamp,
            "phase_before": self.phase_before,
            "phase_after_intent": self.phase_after_intent,
            "reason": self.reason,
            "detail": self.detail,
            "value": self.value,
            "valuation_date": self.valuation_date,
            "valuation_available_at": self.valuation_available_at,
            "valuation_lag": self.valuation_lag,
            "L1": self.thresholds.get("L1"),
            "L2": self.thresholds.get("L2"),
            "L3": self.thresholds.get("L3"),
            "S": self.thresholds.get("S"),
            "allowed_steps": self.allowed_steps,
            "buy_steps_before": self.buy_steps_before,
            "buy_steps_after_intent": self.buy_steps_after_intent,
            "orders": ";".join(f"{o.side}:{o.symbol}:{o.reason}" for o in self.orders),
            "skips": ";".join(self.skips),
            "checks": ";".join(f"{k}={v}" for k, v in sorted(self.checks.items())),
        }


# ------------------------------------------------------------------ 工具
def allowed_steps(value: Optional[float], thresholds: Mapping[str, float]) -> int:
    if value is None or not np.isfinite(value) or value <= 0:
        return 0
    if value <= thresholds["L3"]:
        return 3
    if value <= thresholds["L2"]:
        return 2
    if value <= thresholds["L1"]:
        return 1
    return 0


def _fee_estimate(notional: float, commission_rate: float, min_commission: float) -> float:
    return max(min_commission, notional * commission_rate)


def _cash_floor_sell_qty(
    gap: float,
    price: float,
    commission_rate: float,
    min_commission: float,
    lot_size: int = 100,
) -> int:
    """按佣金精确估算达到现金下限所需卖量，向上取整到整手。"""
    if gap <= 0 or price <= 0:
        return 0
    required_notional = float(gap)
    for _ in range(8):
        shares = int(np.ceil(required_notional / price / lot_size) * lot_size)
        fee = _fee_estimate(shares * price, commission_rate, min_commission)
        net_proceeds = shares * price - fee
        if net_proceeds + 1e-9 >= gap:
            return shares
        required_notional += gap - net_proceeds
    return int(np.ceil(required_notional / price / lot_size) * lot_size)


def _risk_trim(
    ctx: SignalContext,
    symbol: str,
    held: int,
) -> tuple[int, List[str]]:
    """计算单个持仓标的为满足总权益上限和现金下限所需的合并卖量。"""
    account = ctx.account
    price = float(account.prices.get(symbol, 0.0))
    if held <= 0 or price <= 0 or not np.isfinite(price):
        return 0, []
    reasons: List[str] = []
    sell_qty = 0
    if account.equity_weight > ctx.w_cap + ctx.equity_cap_tolerance:
        target_value = ctx.w_cap * account.nav
        excess = account.equity_value - target_value
        if excess > 0:
            reasons.append("risk_equity_cap")
            sell_qty = max(
                sell_qty,
                int(np.ceil(excess / price / 100.0) * 100),
            )
    if account.cash_share < ctx.min_cash_share:
        gap = ctx.min_cash_share * account.nav - account.cash
        if gap > 0:
            reasons.append("risk_cash_floor")
            sell_qty = max(
                sell_qty,
                _cash_floor_sell_qty(
                    gap,
                    price,
                    ctx.commission_rate,
                    ctx.min_commission,
                ),
            )
    return min(max(0, sell_qty), held), reasons


def ratio_gate_series(data: ResearchData, session: pd.Timestamp, base_date,
                      threshold: float = 1.0) -> Optional[Dict[str, object]]:
    """固定基日归一化的股债比较门槛; 只控制新增买入。"""
    frame = data.ratio_signals
    if frame.empty:
        return None
    base = frame[frame["trade_date"] == pd.Timestamp(base_date)]
    if base.empty:
        return None
    tr_base = float(base.iloc[0]["tr300"])
    bond_base = float(base.iloc[0]["bond_index"])
    if tr_base <= 0 or bond_base <= 0:
        return None
    obs = data.ratio_asof(session)
    if obs is None:
        return None
    tr_ratio = obs["tr300"] / tr_base
    bond_ratio = obs["bond_index"] / bond_base
    if bond_ratio <= 0:
        return None
    value = tr_ratio / bond_ratio
    return {
        "value": value,
        "gate_open": bool(value <= threshold),
        "tr300": obs["tr300"],
        "bond_index": obs["bond_index"],
        "trade_date": str(obs["trade_date"].date()),
        "lag_sessions": obs["lag_sessions"],
    }


def _fresh(obs: Optional[Mapping[str, object]], limit: int) -> bool:
    if obs is None:
        return False
    lag = obs.get("lag_sessions")
    return lag is not None and lag <= limit


# ------------------------------------------------------------------ 主策略决策
def evaluate_weekly(ctx: SignalContext) -> Decision:
    """一条主策略账户的周度决策(不含红利模块的独立买入)。"""
    state = ctx.state
    account = ctx.account
    symbol = ctx.primary_symbol
    held = int(account.quantities.get(symbol, 0))
    price = float(account.prices.get(symbol, 0.0))
    stamp = session_ts(ctx.session)
    thresholds = ctx.snapshot.thresholds() if ctx.snapshot else {}
    value = float(ctx.valuation["value"]) if ctx.valuation else None
    valuation_date = str(ctx.valuation["trade_date"].date()) if ctx.valuation else None
    available_at = str(pd.Timestamp(ctx.valuation["available_at"])) if ctx.valuation else None
    lag = int(ctx.valuation["lag_sessions"]) if ctx.valuation else None
    steps = allowed_steps(value, thresholds) if thresholds else 0
    skips: List[str] = []
    checks: Dict[str, object] = {}

    decision = Decision(
        strategy=ctx.strategy,
        session=str(pd.Timestamp(ctx.session).date()),
        timestamp=str(stamp),
        phase_before=state.phase,
        phase_after_intent=state.phase,
        reason="hold",
        detail="",
        value=value,
        valuation_date=valuation_date,
        valuation_available_at=available_at,
        valuation_lag=lag,
        thresholds=thresholds,
        allowed_steps=steps,
        buy_steps_before=state.buy_steps,
        buy_steps_after_intent=state.buy_steps,
    )

    def sell_all(reason: str, detail: str) -> Decision:
        decision.orders.append(OrderIntent(side="sell", symbol=symbol, reason=reason, requested_qty=None,
                                           note="清仓意图锁定至实际份额归零"))
        decision.phase_after_intent = "exiting"
        if state.exit_reason is None:
            state.exit_reason = reason
        state.phase = "exiting"
        decision.reason = reason
        decision.detail = detail
        return decision

    # 1) 已有退出意图
    if state.phase == "exiting" and held > 0:
        return sell_all("exit_intent_repeat", f"未完成退出意图继续, 持有 {held} 份")

    if state.phase == "exiting" and held == 0:
        state.reset_round()
        decision.phase_after_intent = state.phase
        decision.buy_steps_after_intent = state.buy_steps
        decision.checks["round_reset"] = 1

    # 2) 资格退出
    eligibility = ctx.data.eligibility(symbol, ctx.session)
    checks["eligible"] = int(bool(eligibility["eligible"]))
    if held > 0 and not eligibility["eligible"]:
        return sell_all("eligibility_exit", f"产品资格不合格: {eligibility['reason']}")

    # 3) 高估清仓
    if ctx.snapshot is None:
        skips.append("calibration_unavailable")
    if held > 0 and ctx.snapshot is not None and _fresh(ctx.valuation, ctx.max_valuation_lag_sessions) \
            and value is not None and value >= thresholds["S"]:
        reason = "annual_recalibration_exit" if ctx.recalibrated else "overvaluation_exit"
        return sell_all(reason, f"估值 {value:.4f} >= 清仓线 {thresholds['S']:.4f}")

    # 4) 数据有效性
    valuation_ok = bool(ctx.valuation) and np.isfinite(value or np.nan) and (value or 0) > 0 \
        and _fresh(ctx.valuation, ctx.max_valuation_lag_sessions)
    checks["valuation_fresh"] = int(valuation_ok)
    if not valuation_ok:
        skips.append("valuation_missing_or_stale")

    # 5) 风险限额修正
    sell_qty, risk_reasons = _risk_trim(ctx, symbol, held)
    if sell_qty > 0:
        decision.orders.append(OrderIntent(
            side="sell", symbol=symbol, reason=risk_reasons[0], requested_qty=sell_qty,
            note=(
                f"风险修正({'+'.join(risk_reasons)}), 目标权益 {ctx.w_cap:.0%}, "
                f"现金下限 {ctx.min_cash_share:.0%}"
            ),
        ))
        decision.reason = risk_reasons[0]
        decision.detail = f"风险修正卖出 {sell_qty} 份"
        decision.checks["risk_sell_qty"] = sell_qty
        decision.checks["risk_reasons"] = "+".join(risk_reasons)
        return decision

    # 6) 低估分批买入
    if not valuation_ok:
        decision.reason = "skip_no_buy_data_invalid"
        decision.detail = "估值缺失/失效, 不新增买入"
        return decision
    if ctx.snapshot is None:
        failure = ctx.calibration_failure or {}
        decision.reason = "skip_calibration_blocked"
        decision.detail = f"年度校准不可用: {failure.get('reason', 'unknown')}"
        return decision

    if ctx.gate_required:
        gate = ctx.ratio_gate
        if gate is None:
            skips.append("ratio_gate_data_missing")
            decision.reason = "skip_ratio_gate_blocked"
            decision.detail = "股债比较序列缺失或过期, 本周禁止新增买入"
            return decision
        if gate.get("lag_sessions", 99) > ctx.max_valuation_lag_sessions:
            skips.append("ratio_gate_stale")
            decision.reason = "skip_ratio_gate_blocked"
            decision.detail = "股债比较序列超过1个交易日"
            return decision
        decision.checks["ratio_gate"] = round(float(gate["value"]), 6)
        if not gate["gate_open"]:
            decision.reason = "skip_ratio_gate_closed"
            decision.detail = f"股债比较 X={gate['value']:.4f} > 1.0"
            return decision

    if steps == 0:
        decision.reason = "hold_outside_buy_band"
        decision.detail = f"估值 {value:.4f} 高于最高买入线 {thresholds['L1']:.4f}"
        return decision
    if state.buy_steps >= min(steps, 3):
        decision.reason = "hold_steps_exhausted"
        decision.detail = f"已成功买入 {state.buy_steps} 批, 当前档位允许 {steps} 批"
        return decision
    if state.last_buy_session is not None:
        gap_sessions = int(ctx.data.session_pos(ctx.session) - ctx.data.session_pos(state.last_buy_session))
        decision.checks["sessions_since_last_buy"] = gap_sessions
        if gap_sessions < ctx.cooldown_sessions:
            decision.reason = "hold_cooldown"
            decision.detail = f"距上次成功买入 {gap_sessions} 个交易日 < {ctx.cooldown_sessions}"
            return decision

    next_step = state.buy_steps + 1
    target_weight = ctx.w_cap * next_step / 3.0
    current_value = account.value(symbol)
    buy_value = max(0.0, target_weight * account.nav - current_value)
    budget = min(buy_value, ctx.buy_budget_nav_share * account.nav)
    spendable = account.cash - ctx.min_cash_share * account.nav
    budget = min(budget, max(0.0, spendable))
    trade_price = account.trade_price(symbol)
    if trade_price <= 0 or not np.isfinite(trade_price):
        decision.reason = "skip_no_price"
        decision.detail = "信号日无有效收盘价, 不生成买单"
        return decision
    fee_reserve = _fee_estimate(budget, ctx.commission_rate, ctx.min_commission)
    lots = int(np.floor(max(0.0, budget - fee_reserve) / trade_price / 100.0)) * 100
    if lots <= 0:
        decision.reason = "skip_buy_budget_too_small"
        decision.detail = f"可用预算 {budget:.2f} 元不足1手(100份), 不生成买单"
        return decision
    decision.orders.append(OrderIntent(
        side="buy", symbol=symbol, reason=f"buy_step_{next_step}",
        requested_qty=lots, target_weight=target_weight,
        note=f"档位允许{steps}批, 目标权重{target_weight:.4f}",
    ))
    decision.reason = f"buy_step_{next_step}_requested"
    decision.detail = f"估值 {value:.4f} <= L{4 - steps} 阈值, 申请第 {next_step} 批 {lots} 份"
    decision.buy_steps_after_intent = next_step
    decision.phase_after_intent = "holding"
    return decision


# ------------------------------------------------------------------ 红利低波模块
def evaluate_dividend_module(ctx: SignalContext) -> Decision:
    """B 空仓时的红利低波决策; 只处理红利标的自身的买卖。"""
    symbol = ctx.dividend_symbol or "SH512890"
    module_state = ctx.dividend_state
    yield_obs = ctx.dividend_yield
    stamp = session_ts(ctx.session)
    decision = Decision(
        strategy="B_pb_dividend",
        session=str(pd.Timestamp(ctx.session).date()),
        timestamp=str(stamp),
        phase_before=module_state.phase if module_state is not None else "flat",
        phase_after_intent="flat",
        reason="hold",
        detail="",
        value=float(yield_obs["yield_ttm"]) if yield_obs else None,
        valuation_date=str(yield_obs["trade_date"].date()) if yield_obs else None,
        valuation_available_at=str(pd.Timestamp(yield_obs["available_at"])) if yield_obs else None,
        valuation_lag=int(yield_obs["lag_sessions"]) if yield_obs else None,
        thresholds={},
        allowed_steps=0,
        buy_steps_before=0,
        buy_steps_after_intent=0,
    )
    held = int(ctx.account.quantities.get(symbol, 0))

    def sell_all(reason: str, detail: str) -> Decision:
        decision.orders.append(OrderIntent(
            side="sell", symbol=symbol, reason=reason, requested_qty=None,
            note="红利低波清仓意图锁定至实际份额归零",
        ))
        decision.reason = reason
        decision.detail = detail
        decision.phase_after_intent = "exiting"
        if module_state is not None:
            module_state.phase = "exiting"
            if module_state.exit_reason is None:
                module_state.exit_reason = reason
        return decision

    def risk_trim() -> bool:
        sell_qty, reasons = _risk_trim(ctx, symbol, held)
        if sell_qty <= 0:
            return False
        decision.orders.append(OrderIntent(
            side="sell", symbol=symbol, reason=reasons[0], requested_qty=sell_qty,
            note=(
                f"红利低波风险修正({'+'.join(reasons)}), 目标总权益 {ctx.w_cap:.0%}, "
                f"现金下限 {ctx.min_cash_share:.0%}"
            ),
        ))
        decision.reason = reasons[0]
        decision.detail = f"红利低波风险修正卖出 {sell_qty} 份"
        decision.checks["risk_sell_qty"] = sell_qty
        decision.checks["risk_reasons"] = "+".join(reasons)
        return True

    if held > 0 and module_state is not None and module_state.phase == "exiting":
        return sell_all("dividend_exit_intent_repeat", f"未完成红利低波退出意图继续, 持有 {held} 份")
    if held == 0 and module_state is not None and module_state.phase == "exiting":
        module_state.reset_round()
        decision.phase_after_intent = "flat"
        decision.checks["round_reset"] = 1

    eligibility = ctx.data.eligibility(symbol, ctx.session)
    decision.checks["eligible"] = int(bool(eligibility["eligible"]))
    if held > 0 and not eligibility["eligible"]:
        return sell_all("eligibility_exit", f"红利低波产品资格不合格: {eligibility['reason']}")
    if held == 0 and not eligibility["eligible"]:
        decision.reason = "skip_dividend_product_ineligible"
        decision.detail = f"红利低波产品当时不可交易: {eligibility['reason']}"
        return decision

    yield_value = None
    yield_valid = False
    if yield_obs is not None:
        try:
            yield_value = float(yield_obs["yield_ttm"])
            yield_valid = bool(np.isfinite(yield_value) and yield_value >= 0
                               and _fresh(yield_obs, ctx.max_valuation_lag_sessions))
        except (KeyError, TypeError, ValueError):
            yield_valid = False
    decision.checks["yield_fresh"] = int(yield_valid)

    if yield_valid and yield_value is not None and yield_value <= 0.05:
        if held > 0:
            return sell_all("dividend_yield_exit", f"股息率 {yield_value:.4%} <= 5%")
        decision.reason = "hold_dividend_gate_closed"
        decision.detail = f"股息率 {yield_value:.4%} <= 5%, 不买入"
        return decision

    if held > 0 and risk_trim():
        return decision

    if not yield_valid:
        decision.reason = "skip_dividend_data_missing"
        decision.detail = "指数股息率缺失、无效或过期, 禁止新增红利且不制造新退出"
        if held > 0:
            decision.detail += "; 已持有则继续持有并标记数据失效"
        return decision

    y = float(yield_value)
    if held > 0:
        decision.reason = "hold_dividend_position"
        decision.detail = f"股息率 {y:.4%} > 5%, 持有不补足"
        return decision
    if not ctx.dividend_gate_open:
        decision.reason = "hold_dividend_gate_closed"
        decision.detail = "创业板持仓/买入优先或数据无效, 本周不新增红利"
        return decision
    target_weight = 0.30
    budget = min(target_weight * ctx.account.nav, ctx.buy_budget_nav_share * ctx.account.nav)
    spendable = ctx.account.cash - ctx.min_cash_share * ctx.account.nav
    budget = min(budget, max(0.0, spendable))
    price = ctx.account.trade_price(symbol)
    if price <= 0:
        decision.reason = "skip_no_price"
        decision.detail = "红利标的本周无有效收盘价"
        return decision
    fee = _fee_estimate(budget, ctx.commission_rate, ctx.min_commission)
    lots = int(np.floor(max(0.0, budget - fee) / price / 100.0)) * 100
    if lots <= 0:
        decision.reason = "skip_buy_budget_too_small"
        decision.detail = "红利低波可用预算不足1手"
        return decision
    decision.orders.append(OrderIntent(side="buy", symbol=symbol, reason="dividend_yield_entry",
                                       requested_qty=lots, target_weight=target_weight,
                                       note="股息率>5%, 申请30%目标仓位"))
    decision.reason = "dividend_yield_entry_requested"
    decision.detail = f"股息率 {y:.4%} > 5%, 申请 {lots} 份"
    decision.phase_after_intent = "holding"
    return decision
