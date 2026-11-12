"""周度状态机测试: 优先级、档位边界、冷却、风险限额、股债门槛与红利切换。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.valuation_bands import fixtures
from research.valuation_bands.signals import (
    AccountSnapshot,
    SignalContext,
    StrategyState,
    allowed_steps,
    evaluate_dividend_module,
    evaluate_weekly,
)

START = "2015-01-01"
END = "2016-12-31"


def base_data(amount: float = 2e8, volume: float = 1e7, tail_zero: int = 0):
    calendar = pd.DatetimeIndex(
        __import__("research.valuation_bands.data", fromlist=["x"]).build_calendar(START, END))
    n = len(calendar)
    close = np.full(n, 4.0)
    amounts = np.full(n, amount)
    volumes = np.full(n, volume)
    if tail_zero:
        amounts[-tail_zero:] = 0.0
        volumes[-tail_zero:] = 0.0
    return fixtures.make_dataset(
        START, END,
        price_map={"000300": np.full(n, 3000.0)},
        valuation_map={("000300", "PE_TTM"): np.full(n, 12.0)},
        etf_map={
            "SH510300": {"close": close, "open": close, "amount": amounts, "volume": volumes},
            "SH512890": {
                "close": np.ones(n), "open": np.ones(n),
                "amount": np.full(n, 2e8), "volume": np.full(n, 1e7),
            },
        },
    )


def signal_session(data, position=None):
    calendar = data.calendar
    return pd.Timestamp(calendar[position if position is not None else len(calendar) - 5])


def valuation(value: float, lag: int = 1, session=None) -> dict:
    session = pd.Timestamp(session) if session is not None else pd.Timestamp("2016-12-20")
    return {
        "trade_date": session - pd.Timedelta(days=3),
        "value": float(value),
        "available_at": pd.Timestamp(session),
        "availability_basis": "assumed_next_session_19",
        "source_id": "fixture",
        "lag_sessions": lag,
    }


def make_ctx(
    *,
    value=10.0,
    lag=1,
    nav=1_000_000.0,
    cash=1_000_000.0,
    quantity=0,
    price=4.0,
    l1=11.0,
    l2=10.0,
    l3=9.0,
    s=16.0,
    w_cap=0.6,
    state=None,
    cooldown=20,
    last_buy_session=None,
    buy_steps=0,
    phase="flat",
    gate_required=False,
    gate=None,
    ratio_obs=None,
    data=None,
    session=None,
    dividend_yield=None,
    dividend_gate_open=False,
    recalibrated=False,
):
    data = data if data is not None else base_data()
    session = signal_session(data) if session is None else pd.Timestamp(session)
    account = AccountSnapshot(
        nav=nav, cash=cash, dividend_receivable=0.0,
        quantities={"SH510300": quantity}, prices={"SH510300": price},
    )
    state = state or StrategyState()
    state.phase = phase
    state.buy_steps = buy_steps
    state.last_buy_session = pd.Timestamp(last_buy_session) if last_buy_session is not None else None
    obs = None if value is None else valuation(value, lag, session)
    return SignalContext(
        strategy="A", session=session, data=data,
        snapshot=fixtures.make_snapshot(l1, l2, l3, s),
        calibration_failure=None, valuation=obs, account=account, state=state,
        primary_symbol="SH510300", w_cap=w_cap, cooldown_sessions=cooldown, exit_q=0.85,
        ratio_gate=gate if gate is not None else ratio_obs, gate_required=gate_required,
        dividend_yield=dividend_yield, dividend_symbol="SH512890",
        dividend_gate_open=dividend_gate_open, recalibrated=recalibrated,
    )


class BandTests(unittest.TestCase):
    def test_allowed_steps_boundaries(self):
        thresholds = {"L1": 11.0, "L2": 10.0, "L3": 9.0, "S": 16.0}
        self.assertEqual(allowed_steps(9.0, thresholds), 3)
        self.assertEqual(allowed_steps(8.999, thresholds), 3)
        self.assertEqual(allowed_steps(9.001, thresholds), 2)
        self.assertEqual(allowed_steps(10.0, thresholds), 2)
        self.assertEqual(allowed_steps(10.001, thresholds), 1)
        self.assertEqual(allowed_steps(11.0, thresholds), 1)
        self.assertEqual(allowed_steps(11.001, thresholds), 0)
        self.assertEqual(allowed_steps(None, thresholds), 0)
        self.assertEqual(allowed_steps(-1.0, thresholds), 0)
        self.assertEqual(allowed_steps(float("inf"), thresholds), 0)


class ExitPriorityTests(unittest.TestCase):
    def test_existing_exit_intent_not_cancelled_by_low_valuation(self):
        ctx = make_ctx(value=5.0, quantity=10_000, phase="exiting")
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "exit_intent_repeat")
        self.assertEqual(decision.orders[0].side, "sell")
        self.assertIsNone(decision.orders[0].requested_qty)

    def test_overvaluation_exit_and_partial_then_zero(self):
        ctx = make_ctx(value=16.5, quantity=10_000)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "overvaluation_exit")
        self.assertEqual(decision.phase_after_intent, "exiting")
        self.assertEqual(ctx.state.phase, "exiting")

    def test_recalibration_exit_labelled(self):
        ctx = make_ctx(value=16.5, quantity=10_000, recalibrated=True)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "annual_recalibration_exit")

    def test_no_exit_without_valid_valuation(self):
        ctx = make_ctx(value=None, quantity=10_000)
        decision = evaluate_weekly(ctx)
        self.assertNotIn(decision.reason, ("overvaluation_exit", "annual_recalibration_exit"))
        self.assertEqual(decision.orders, [])

    def test_stale_valuation_does_not_trigger_new_exit(self):
        ctx = make_ctx(value=20.0, lag=2, quantity=10_000)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "skip_no_buy_data_invalid")

    def test_repeat_exit_preserves_original_lock_reason(self):
        ctx = make_ctx(value=5.0, quantity=10_000, phase="exiting")
        ctx.state.exit_reason = "overvaluation_exit"
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "exit_intent_repeat")
        self.assertEqual(ctx.state.exit_reason, "overvaluation_exit")

    def test_flat_round_reset_updates_phase_after_intent(self):
        ctx = make_ctx(value=12.0, quantity=0, phase="exiting", buy_steps=2)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.phase_before, "exiting")
        self.assertEqual(decision.phase_after_intent, "flat")
        self.assertEqual(decision.buy_steps_after_intent, 0)

    def test_eligibility_exit(self):
        data = base_data(tail_zero=30)
        ctx = make_ctx(quantity=10_000, data=data, session=signal_session(data))
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "eligibility_exit")
        self.assertEqual(decision.orders[0].side, "sell")

    def test_flat_high_valuation_creates_no_fake_exit(self):
        ctx = make_ctx(value=20.0, quantity=0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.orders, [])
        self.assertEqual(decision.reason, "hold_outside_buy_band")


class BuyTests(unittest.TestCase):
    def test_first_deep_band_buys_only_one_batch(self):
        ctx = make_ctx(value=8.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.allowed_steps, 3)
        self.assertEqual(len(decision.orders), 1)
        self.assertEqual(decision.orders[0].reason, "buy_step_1")
        self.assertEqual(decision.buy_steps_after_intent, 1)
        self.assertAlmostEqual(decision.orders[0].target_weight, 0.2, places=12)
        self.assertEqual(decision.orders[0].requested_qty, 49_900)

    def test_buy_respects_thirty_percent_budget(self):
        # 第二批目标 40% 净值(400000) > 单次买入预算 30%(300000), 预算生效
        ctx = make_ctx(value=8.0, buy_steps=1, phase="holding", quantity=0,
                       cash=1_000_000.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        lots = decision.orders[0].requested_qty
        self.assertLessEqual(lots * 4.0, 300_000 + 1e-9)
        self.assertGreater(lots * 4.0, 299_000)

    def test_cash_floor_blocks_oversized_buy(self):
        ctx = make_ctx(value=8.0, cash=50_200.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "skip_buy_budget_too_small")
        self.assertEqual(decision.orders, [])

    def test_steps_exhausted(self):
        ctx = make_ctx(value=8.0, buy_steps=3, phase="holding", quantity=120_000,
                       cash=400_000.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "hold_steps_exhausted")

    def test_allowed_steps_caps_batch(self):
        # 估值只在第一档 -> 只允许 1 批; 已买 1 批后不再加仓
        ctx = make_ctx(value=10.8, buy_steps=1, phase="holding", quantity=50_000,
                       cash=800_000.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "hold_steps_exhausted")

    def test_cooldown_19_blocked_20_allowed(self):
        data = base_data()
        session = signal_session(data)
        position = data.session_pos(session)
        before = data.calendar[position - 19]
        ctx = make_ctx(value=9.5, buy_steps=1, phase="holding", quantity=50_000, cash=800_000.0,
                       nav=1_000_000.0, last_buy_session=before, data=data, session=session)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "hold_cooldown")
        exact = data.calendar[position - 20]
        ctx = make_ctx(value=9.5, buy_steps=1, phase="holding", quantity=50_000, cash=800_000.0,
                       nav=1_000_000.0, last_buy_session=exact, data=data, session=session)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.orders[0].reason, "buy_step_2")
        self.assertAlmostEqual(decision.orders[0].target_weight, 0.4, places=12)

    def test_underweight_position_still_buys_next_step(self):
        # 部分成交导致实际仓位低于理论目标时, 下一批仍按目标权重计算
        ctx = make_ctx(value=8.0, buy_steps=1, phase="holding", quantity=20_000,
                       cash=900_000.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.orders[0].reason, "buy_step_2")
        self.assertAlmostEqual(decision.orders[0].target_weight, 0.4, places=12)

    def test_missing_signal_trade_price_blocks_buy_even_with_carry_valuation(self):
        ctx = make_ctx(value=8.0)
        ctx.account.trade_prices = {"SH510300": 0.0}
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "skip_no_price")
        self.assertEqual(decision.orders, [])


class RiskTests(unittest.TestCase):
    def test_equity_cap_trim(self):
        # 640000/1000000 = 64% > 60% + 3pp
        ctx = make_ctx(value=12.0, quantity=160_000, cash=360_000.0, nav=1_000_000.0)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "risk_equity_cap")
        self.assertEqual(decision.orders[0].side, "sell")
        self.assertEqual(decision.orders[0].requested_qty, 10_000)

    def test_cash_floor_trim(self):
        # 权益 60%(未超上限) + 现金 3% + 分红应收 37%; 只有现金下限被触发
        ctx = make_ctx(value=12.0, quantity=150_000, cash=30_000.0, nav=1_000_000.0)
        ctx.account.dividend_receivable = 370_000.0
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "risk_cash_floor")
        self.assertEqual(decision.orders[0].requested_qty, 5_100)

    def test_combined_risk_conditions_emit_one_max_size_sell(self):
        ctx = make_ctx(value=12.0, quantity=200_000, cash=30_000.0, nav=1_000_000.0)
        ctx.account.dividend_receivable = 170_000.0
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "risk_equity_cap")
        self.assertEqual(len(decision.orders), 1)
        self.assertEqual(decision.orders[0].requested_qty, 50_000)
        self.assertEqual(decision.checks["risk_reasons"], "risk_equity_cap+risk_cash_floor")

    def test_risk_trim_does_not_reset_buy_steps(self):
        ctx = make_ctx(value=12.0, quantity=160_000, cash=360_000.0, nav=1_000_000.0,
                       buy_steps=2, phase="holding")
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "risk_equity_cap")
        self.assertEqual(ctx.state.buy_steps, 2)
        self.assertEqual(decision.phase_after_intent, "holding")


class GateTests(unittest.TestCase):
    def test_gate_blocks_buy(self):
        gate = {"value": 1.05, "gate_open": False, "tr300": 1.0, "bond_index": 1.0,
                "trade_date": "2016-12-20", "lag_sessions": 1}
        ctx = make_ctx(value=8.0, gate_required=True, gate=gate)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "skip_ratio_gate_closed")
        self.assertEqual(decision.orders, [])

    def test_gate_missing_blocks_buy(self):
        ctx = make_ctx(value=8.0, gate_required=True, gate=None)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.reason, "skip_ratio_gate_blocked")

    def test_gate_open_allows_buy(self):
        gate = {"value": 0.95, "gate_open": True, "tr300": 1.0, "bond_index": 1.0,
                "trade_date": "2016-12-20", "lag_sessions": 1}
        ctx = make_ctx(value=8.0, gate_required=True, gate=gate)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.orders[0].reason, "buy_step_1")

    def test_gate_never_triggers_sell(self):
        gate = {"value": 1.5, "gate_open": False, "tr300": 1.0, "bond_index": 1.0,
                "trade_date": "2016-12-20", "lag_sessions": 1}
        ctx = make_ctx(value=12.0, quantity=50_000, phase="holding", cash=800_000.0,
                       nav=1_000_000.0, gate_required=True, gate=gate)
        decision = evaluate_weekly(ctx)
        self.assertEqual(decision.orders, [])


class DividendModuleTests(unittest.TestCase):
    def dividend_ctx(self, yield_value, held=0, gate_open=True, missing=False, lag=1,
                     cash=1_000_000.0, nav=1_000_000.0, dividend_state=None):
        data = base_data()
        session = signal_session(data)
        account = AccountSnapshot(
            nav=nav, cash=cash, dividend_receivable=0.0,
            quantities={"SZ159915": 0, "SH512890": held},
            prices={"SZ159915": 2.0, "SH512890": 1.0},
        )
        obs = None if missing else {
            "trade_date": session - pd.Timedelta(days=3), "yield_ttm": float(yield_value),
            "available_at": session, "lag_sessions": lag,
        }
        return SignalContext(
            strategy="B", session=session, data=data, snapshot=fixtures.make_snapshot(4.4, 4.0, 3.6, 6.0),
            calibration_failure=None, valuation=None, account=account, state=StrategyState(),
            primary_symbol="SZ159915", w_cap=0.4, cooldown_sessions=20, exit_q=0.75,
            dividend_yield=obs, dividend_symbol="SH512890", dividend_gate_open=gate_open,
            dividend_state=dividend_state,
        )

    def test_yield_above_threshold_buys_thirty_percent(self):
        ctx = self.dividend_ctx(0.052)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "dividend_yield_entry_requested")
        self.assertAlmostEqual(decision.orders[0].target_weight, 0.30, places=12)
        self.assertLessEqual(decision.orders[0].requested_qty * 1.0, 300_000)

    def test_yield_exactly_five_percent_exits_and_does_not_buy(self):
        ctx = self.dividend_ctx(0.05, held=0)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.orders, [])
        ctx = self.dividend_ctx(0.05, held=100_000)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "dividend_yield_exit")
        self.assertEqual(decision.orders[0].side, "sell")

    def test_missing_yield_blocks_new_buy(self):
        ctx = self.dividend_ctx(0.0, missing=True)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "skip_dividend_data_missing")
        self.assertEqual(decision.orders, [])

    def test_stale_yield_blocks_new_buy(self):
        ctx = self.dividend_ctx(0.06, lag=2)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "skip_dividend_data_missing")
        self.assertEqual(decision.orders, [])

    def test_ineligible_dividend_product_blocks_entry(self):
        ctx = self.dividend_ctx(0.06)
        bars = ctx.data.frames["etf_bars"]
        ctx.data.frames["etf_bars"] = bars[bars["symbol"] != "SH512890"].copy()
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "skip_dividend_product_ineligible")
        self.assertEqual(decision.orders, [])

    def test_holding_not_topped_up(self):
        ctx = self.dividend_ctx(0.06, held=100_000)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "hold_dividend_position")
        self.assertEqual(decision.orders, [])

    def test_gate_closed_no_new_buy(self):
        ctx = self.dividend_ctx(0.06, gate_open=False)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "hold_dividend_gate_closed")
        self.assertEqual(decision.orders, [])

    def test_existing_dividend_exit_survives_missing_yield(self):
        state = StrategyState(phase="exiting", exit_reason="dividend_yield_exit")
        ctx = self.dividend_ctx(0.0, held=100_000, missing=True, dividend_state=state)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "dividend_exit_intent_repeat")
        self.assertEqual(len(decision.orders), 1)
        self.assertEqual(state.phase, "exiting")

    def test_dividend_risk_trim_applies_when_yield_missing(self):
        ctx = self.dividend_ctx(0.0, held=450_000, missing=True,
                                cash=550_000.0, nav=1_000_000.0)
        decision = evaluate_dividend_module(ctx)
        self.assertEqual(decision.reason, "risk_equity_cap")
        self.assertEqual(decision.orders[0].symbol, "SH512890")
        self.assertEqual(decision.orders[0].requested_qty, 50_000)


if __name__ == "__main__":
    unittest.main()
