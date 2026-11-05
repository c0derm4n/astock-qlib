"""Focused tests for the causal MA5/MA10 4% research signal."""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from research.ma_breakout.signals import (
    BreakoutTargetSelector,
    SignalBook,
    SignalParameters,
    bounded_daily_targets,
    build_signal_conditions,
    compute_technical_signals,
    effective_eligibility,
    next_technical_state,
)
from research.ma_breakout.engine import simulate_research_cash
from src.formal_rules import ROOT
from src.formal_cash import audit_cash_ledger


class BreakoutSignalTests(unittest.TestCase):
    def parameters(self):
        return SignalParameters(threshold=.04, entry_window=5, exit_window=10)

    def test_return_threshold_exact_boundary_and_two_day_compounding(self):
        dates = pd.bdate_range("2026-01-01", periods=15)
        prices = pd.DataFrame({"SH515880": 10.}, index=dates)
        prices.iloc[-3:, 0] = [9.8, 10. / 1.04, 10.]
        conditions = build_signal_conditions(prices, self.parameters())
        self.assertAlmostEqual(conditions["r1"].iloc[-1, 0], .04, places=12)
        self.assertTrue(conditions["entry_condition"].iloc[-1, 0])

        prices.iloc[-1, 0] = (10. / 1.04) * (1. + .04 - 1e-9)
        below = build_signal_conditions(prices, self.parameters())
        self.assertFalse(below["entry_condition"].iloc[-1, 0])

        prices.iloc[-3, 0] = 10. / (1. + .04) ** 2
        prices.iloc[-2, 0] = 10. / (1. + .04)
        prices.iloc[-1, 0] = 10.
        compounding = build_signal_conditions(prices, self.parameters())
        self.assertAlmostEqual(
            compounding["r2"].iloc[-1, 0], (1. + .04) ** 2 - 1, places=12
        )
        self.assertTrue(compounding["entry_condition"].iloc[-1, 0])

    def test_ma_entry_equality_and_ma_exit_equality(self):
        dates = pd.bdate_range("2026-01-01", periods=12)
        values = [8.8, 8.9, 9., 9.1, 9.2, 9.3,
                  10.384615384615385, 10.384615384615385,
                  9.615384615384615, 9.615384615384615, 10., 10.]
        prices = pd.DataFrame({"SH515880": values}, index=dates)
        conditions = build_signal_conditions(prices, self.parameters())
        self.assertAlmostEqual(
            conditions["ma_entry"].iloc[10, 0], conditions["price"].iloc[10, 0],
            places=12,
        )
        self.assertTrue(conditions["entry_condition"].iloc[10, 0])

        equal_ten = pd.DataFrame({"SH515880": 10.}, index=dates)
        equal_ten.iloc[0, 0] = 9.5
        equal_ten.iloc[1, 0] = 11.5
        exit_conditions = build_signal_conditions(equal_ten, self.parameters())
        self.assertAlmostEqual(
            exit_conditions["ma_exit"].iloc[-1, 0],
            exit_conditions["price"].iloc[-1, 0],
            places=12,
        )
        self.assertFalse(exit_conditions["exit_condition"].iloc[-1, 0])

    def test_exit_priority_and_latched_holding(self):
        self.assertFalse(next_technical_state(True, True, True, True))
        self.assertFalse(next_technical_state(False, True, True, True))
        self.assertFalse(next_technical_state(True, False, False, False))
        self.assertTrue(next_technical_state(True, True, False, False))
        self.assertFalse(next_technical_state(False, True, True, True))

        dates = pd.bdate_range("2026-01-01", periods=20)
        prices = pd.DataFrame({"SH515880": 10.}, index=dates)
        prices.iloc[1:3, 0] = 9.5
        prices.iloc[3:10, 0] = np.linspace(9.9, 10.1, 7)
        prices.iloc[10:, 0] = 10.
        eligible = pd.DataFrame(True, index=dates, columns=prices.columns)
        book, frames = compute_technical_signals(prices, eligible, self.parameters())
        entered = frames["entry_event"].iloc[:, 0]
        held = frames["held"].iloc[:, 0]
        self.assertTrue(entered.any())
        entry_day = entered[entered].index[0]
        later = held.loc[prices.index[10:]]
        self.assertTrue(later.iloc[0])
        self.assertTrue(held.loc[entry_day])

    def test_ineligibility_resets_state_and_prefix_is_future_invariant(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        values = 10. * np.power(1.02, np.arange(len(dates)))
        values[14:] = values[13]
        prices = pd.DataFrame({"SH515880": values}, index=dates)
        eligibility = pd.DataFrame(True, index=dates, columns=prices.columns)
        eligibility.iloc[20:, 0] = False
        _, full = compute_technical_signals(prices, eligibility, self.parameters())
        self.assertTrue(full["held"].iloc[19, 0])
        self.assertFalse(full["held"].iloc[20, 0])
        self.assertTrue(full["exit_trigger"].iloc[20, 0])

        cutoff = dates[14]
        prefix_full = full["held"].loc[:cutoff].copy()
        changed = prices.copy()
        changed.loc[changed.index > cutoff] *= 100
        _, changed_frames = compute_technical_signals(
            changed, eligibility, self.parameters()
        )
        pd.testing.assert_frame_equal(
            prefix_full, changed_frames["held"].loc[:cutoff]
        )

    def test_state_resets_at_experiment_start_despite_longer_warmup(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        values = 10. * np.power(1.02, np.arange(len(dates)))
        values[14:] = values[13]
        prices = pd.DataFrame({"SH515880": values}, index=dates)
        eligibility = pd.DataFrame(True, index=dates, columns=prices.columns)
        state_start = dates[15]
        _, states = compute_technical_signals(
            prices, eligibility, self.parameters(), state_start=state_start
        )
        inherited = compute_technical_signals(
            prices, eligibility, self.parameters()
        )[1]["held"].loc[:state_start]
        self.assertTrue(bool(inherited.iloc[:-1].any().any()))
        self.assertFalse(bool(states["held"].loc[:state_start].any().any()))

    def test_fallback_state_only_accrues_inside_communication_ineligibility(self):
        dates = pd.bdate_range("2026-01-01", periods=22)
        prices = pd.DataFrame({
            "SH515880": 10. * np.power(1.02, np.arange(len(dates))),
            "SH518880": 10.,
            "SH510300": 10. * np.power(1.02, np.arange(len(dates))),
        }, index=dates)
        eligibility = pd.DataFrame(True, index=dates, columns=prices.columns)
        eligibility.loc[dates[:8], "SH515880"] = False
        eligibility.loc[dates[8:14], "SH515880"] = True
        eligibility.loc[dates[14:19], "SH515880"] = False
        effective = effective_eligibility(
            eligibility, "SH515880", "SH510300"
        )
        self.assertTrue(effective.loc[dates[4], "SH510300"])
        self.assertFalse(effective.loc[dates[10], "SH510300"])
        self.assertTrue(effective.loc[dates[18], "SH510300"])

    def synthetic_book(self, rows):
        dates = pd.bdate_range("2026-01-01", periods=3)
        trace_rows = []
        for date in dates:
            for symbol, values in rows.items():
                trace_rows.append({
                    "date": date, "symbol": symbol,
                    "price": 10., "r1": .05, "r2": .05,
                    "ma_entry": 9.9, "ma_exit": 9.8,
                    "eligible": True,
                    "entry_condition": True,
                    "exit_condition": False,
                    "entry_event": True,
                    "exit_trigger": False,
                    "held": values["held"],
                })
        return SignalBook(pd.DataFrame(trace_rows).set_index(["date", "symbol"]))

    def selector_policy(self):
        policy = json.loads(
            (ROOT / "strategy_registry.json").read_text(encoding="utf-8")
        )["strategies"][1]["policy"]
        policy = dict(policy)
        policy.update(rebalance_band=.03, max_daily_turnover=.3)
        return policy

    def test_eligibility_substitution_uses_fallback_own_signal_state(self):
        dates = pd.bdate_range("2026-01-01", periods=3)
        returns = pd.DataFrame(
            {"SH515880": 0., "SH518880": 0., "SH510300": 0.}, index=dates
        )
        book = self.synthetic_book({
            "SH515880": {"held": False},
            "SH518880": {"held": True},
            "SH510300": {"held": True},
        })
        selector = BreakoutTargetSelector(
            book,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        targets = selector(
            pd.Series(dtype=float), ["SH518880", "SH510300"], {},
            {}, returns, self.selector_policy(),
        )
        self.assertEqual(set(targets), {"SH510300", "SH518880"})
        self.assertAlmostEqual(sum(targets.values()), .30)
        self.assertAlmostEqual(
            targets["SH510300"] / targets["SH518880"], .6 / .35
        )

        csi_stopped = self.synthetic_book({
            "SH515880": {"held": False},
            "SH518880": {"held": True},
            "SH510300": {"held": False},
        })
        selector = BreakoutTargetSelector(
            csi_stopped,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        targets = selector(
            pd.Series(dtype=float), ["SH518880", "SH510300"], {},
            {}, returns, self.selector_policy(),
        )
        self.assertNotIn("SH510300", targets)

    def test_technical_stop_does_not_switch_to_csi300(self):
        dates = pd.bdate_range("2026-01-01", periods=3)
        returns = pd.DataFrame(
            {"SH515880": 0., "SH518880": 0., "SH510300": 0.}, index=dates
        )
        book = self.synthetic_book({
            "SH515880": {"held": False},
            "SH518880": {"held": True},
            "SH510300": {"held": True},
        })
        selector = BreakoutTargetSelector(
            book,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        targets = selector(
            pd.Series(dtype=float), ["SH515880", "SH510300", "SH518880"],
            {}, {}, returns, self.selector_policy(),
        )
        self.assertNotIn("SH515880", targets)
        self.assertNotIn("SH510300", targets)

    def test_exiting_position_blocks_reentry_and_drift_band_blocks_pyramiding(self):
        dates = pd.bdate_range("2026-01-01", periods=3)
        returns = pd.DataFrame(
            {"SH515880": 0., "SH518880": 0., "SH510300": 0.}, index=dates
        )
        book = self.synthetic_book({
            "SH515880": {"held": True},
            "SH518880": {"held": True},
            "SH510300": {"held": True},
        })
        selector = BreakoutTargetSelector(
            book,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        targets = selector(
            pd.Series(dtype=float), ["SH515880", "SH518880", "SH510300"],
            {"SH515880": .6, "SH518880": .35}, {},
            returns, self.selector_policy(), exiting={"SH515880"},
        )
        self.assertEqual(targets, {"SH518880": .35})

        selector = BreakoutTargetSelector(
            book,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        current = {"SH515880": .61, "SH518880": .34}
        targets = selector(
            pd.Series(dtype=float), ["SH515880", "SH518880", "SH510300"],
            current, {}, returns, self.selector_policy(),
        )
        self.assertEqual(targets, current)

    def test_exit_target_bypasses_band_and_buy_budget_caps_new_entries(self):
        policy = self.selector_policy()
        targets = bounded_daily_targets(
            {"SH518880": .35}, {"SH515880": .6, "SH518880": .35},
            pd.DataFrame(0., index=pd.bdate_range("2025-01-01", periods=60),
                         columns=["SH515880", "SH518880"]),
            policy,
        )
        self.assertNotIn("SH515880", targets)
        self.assertAlmostEqual(targets["SH518880"], .35)
        initial = bounded_daily_targets(
            {"SH515880": .6, "SH518880": .35}, {},
            pd.DataFrame(0., index=pd.bdate_range("2025-01-01", periods=60),
                         columns=["SH515880", "SH518880"]),
            policy,
        )
        self.assertAlmostEqual(initial.sum(), .3, places=12)

    def test_failed_first_liquidation_keeps_blocking_reentry_until_fresh_entry(self):
        dates = pd.bdate_range("2026-01-05", periods=10)
        symbols = ("SH510300", "SH515880", "SH518880")
        values = [10., 10., 10., 10., 10., 10., 12., 12., 12., 12.]
        rows = []
        for symbol in symbols:
            for date, value in zip(dates, values):
                rows.append({
                    "date": date, "symbol": symbol,
                    "open": value, "close": value,
                    "adj_open": value, "adj_close": value,
                    "volume": 1e8, "adj_factor": 1.,
                })
        bars = pd.DataFrame(rows)
        market = type("SyntheticMarket", (), {})()
        market.bars = bars
        market.calendar = dates
        market.eligibility = bars[["date", "symbol"]].assign(
            eligible=True, avg_amount=1e9
        )
        market.wide = lambda field: market.bars.pivot(
            index="date", columns="symbol", values=field
        ).reindex(dates)

        trace_rows = []
        state_by_day = {
            1: (True, True, False),
            2: (True, False, False),
            3: (True, False, False),
            4: (True, False, False),
            5: (False, False, True),
            6: (False, False, False),
            7: (True, True, False),
            8: (True, False, False),
            9: (True, False, False),
        }
        for index, (held, entered, exited) in state_by_day.items():
            for symbol in symbols:
                trace_rows.append({
                    "date": dates[index], "symbol": symbol,
                    "price": values[index], "r1": .05, "r2": .05,
                    "ma_entry": 9.9, "ma_exit": 9.8,
                    "eligible": True,
                    "entry_condition": entered,
                    "exit_condition": exited,
                    "entry_event": entered if symbol == "SH515880" else False,
                    "exit_trigger": exited if symbol == "SH515880" else False,
                    "held": held if symbol == "SH515880" else False,
                })
        book = SignalBook(
            pd.DataFrame(trace_rows).set_index(["date", "symbol"])
        )
        candidate = {
            "name": "synthetic-breakout",
            "family": "fixed",
            "weights": {"SH515880": .6, "SH518880": .35},
            "fallbacks": {"SH515880": {"SH510300": 1.}},
        }
        policy = self.selector_policy() | {"initial_capital": 1_000_000.}
        selector = BreakoutTargetSelector(
            book, candidate, communication="SH515880",
            fallback="SH510300", gold="SH518880", volatility_target=None,
        )
        report, trades, actions, _, issues, order_log = simulate_research_cash(
            market, candidate, policy, dates[0], (),
            signal_dates=dates[1:], execution_delay_sessions=2,
            selector=selector,
        )
        audit_cash_ledger(
            market, report, trades, actions, policy["initial_capital"]
        )
        sells = order_log[order_log.side.eq("sell")].reset_index(drop=True)
        self.assertEqual(list(sells.status), ["unfilled", "filled"])
        self.assertEqual(list(sells.reason), ["price_limit", None])
        self.assertEqual(list(sells.last_attempt_date), [
            str(dates[7].date()), str(dates[9].date()),
        ])
        self.assertEqual(list(sells.fill_date), [None, str(dates[9].date())])
        state_at_rebound = next(
            row for row in selector.trace if row["date"] == str(dates[6].date())
        )
        self.assertEqual(state_at_rebound["target_weights"], {})
        self.assertIn(
            "SH515880", state_at_rebound["blocked_until_post_clear_entry"]
        )

    def test_post_clear_entry_condition_unlocks_without_new_state_transition(self):
        dates = pd.bdate_range("2026-01-05", periods=4)
        symbols = ("SH510300", "SH515880", "SH518880")
        trace_rows = []
        scenarios = {
            dates[0]: (True, False, True, True),
            dates[1]: (True, True, False, True),
            dates[2]: (True, True, False, False),
        }
        for date, (held, entry_condition, exit_condition, entry_event) in scenarios.items():
            for symbol in symbols:
                trace_rows.append({
                    "date": date, "symbol": symbol,
                    "price": 10., "r1": .05, "r2": .05,
                    "ma_entry": 9.9, "ma_exit": 9.8,
                    "eligible": True,
                    "entry_condition": entry_condition
                    if symbol == "SH515880" else False,
                    "exit_condition": exit_condition
                    if symbol == "SH515880" else False,
                    "entry_event": entry_event
                    if symbol == "SH515880" else False,
                    "exit_trigger": exit_condition
                    if symbol == "SH515880" else False,
                    "held": held if symbol == "SH515880" else False,
                })
        book = SignalBook(pd.DataFrame(trace_rows).set_index(["date", "symbol"]))
        selector = BreakoutTargetSelector(
            book,
            {
                "weights": {"SH515880": .6, "SH518880": .35},
                "fallbacks": {"SH515880": {"SH510300": 1.}},
            },
            communication="SH515880",
            fallback="SH510300",
            gold="SH518880",
            volatility_target=None,
        )
        returns = pd.DataFrame(
            {"SH515880": 0., "SH518880": 0., "SH510300": 0.},
            index=dates,
        )
        policy = self.selector_policy()
        selector(
            pd.Series(dtype=float), ["SH515880", "SH518880", "SH510300"],
            {"SH515880": .6}, {}, returns.loc[:dates[0]], policy,
        )
        blocked = selector(
            pd.Series(dtype=float), ["SH515880", "SH518880", "SH510300"],
            {"SH515880": .6}, {}, returns.loc[:dates[1]], policy,
        )
        self.assertEqual(blocked, {})
        released = selector(
            pd.Series(dtype=float), ["SH515880", "SH518880", "SH510300"],
            {}, {}, returns.loc[:dates[2]], policy,
        )
        self.assertIn("SH515880", released)


if __name__ == "__main__":
    unittest.main()
