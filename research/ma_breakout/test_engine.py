"""Synthetic acceptance tests for the MA breakout cash engine."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.ma_breakout.engine import (
    PRODUCTION_ISSUE_KEYS,
    build_execution_schedule,
    simulate_research_cash,
)
from src.formal_cash import audit_cash_ledger, simulate_cash
from src.formal_rules import ROOT, registry, weekly_sessions


class ResearchCashEngineTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-01-05", periods=10)
        self.policy = json.loads(
            (ROOT / "research/ex_nasdaq/selected_strategy.json").read_text(encoding="utf-8")
        )["policy"]
        self.policy.update(
            initial_capital=100_000.,
            commission=0.,
            minimum_commission=0.,
            slippage=0.,
            impact_coefficient=0.,
            conservative_buy_budget=True,
            max_adv_participation=.02,
        )
        self.candidate = {
            "name": "synthetic-ma-research",
            "family": "fixed",
            "weights": {"SH515880": .5},
        }

    def market(self, opens=None, closes=None, volumes=None):
        symbols = ("SH510300", "SH515880", "SH518880")
        open_values = np.asarray(
            opens if opens is not None else [10.] * len(self.dates), dtype=float
        )
        close_values = np.asarray(
            closes if closes is not None else open_values, dtype=float
        )
        volume_values = np.asarray(
            volumes if volumes is not None else [1e8] * len(self.dates), dtype=float
        )
        self.assertEqual(open_values.shape, (len(self.dates),))
        self.assertEqual(close_values.shape, (len(self.dates),))
        self.assertEqual(volume_values.shape, (len(self.dates),))
        rows = []
        for symbol in symbols:
            for index, date in enumerate(self.dates):
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "open": open_values[index],
                    "close": close_values[index],
                    "adj_open": open_values[index],
                    "adj_close": close_values[index],
                    "volume": volume_values[index],
                    "adj_factor": 1.,
                })
        bars = pd.DataFrame(rows)
        market = SimpleNamespace(
            bars=bars,
            calendar=self.dates,
            eligibility=bars[["date", "symbol"]].assign(
                eligible=True, avg_amount=1e9
            ),
        )
        market.wide = lambda field: market.bars.pivot(
            index="date", columns="symbol", values=field
        ).reindex(self.dates)
        return market

    def run_engine(self, market, targets, actions=(), delay=1, signals=None):
        signals = self.dates[1:4] if signals is None else signals
        if isinstance(targets, dict):
            selector = lambda *args, **kwargs: targets
        else:
            selector = targets
        result = simulate_research_cash(
            market,
            self.candidate,
            self.policy,
            self.dates[0],
            actions,
            signal_dates=signals,
            execution_delay_sessions=delay,
            selector=selector,
        )
        report, trades, action_log, _, _, _ = result
        audit_cash_ledger(
            market, report, trades, action_log, self.policy["initial_capital"]
        )
        return result

    def test_daily_t_plus_1_schedule_uses_explicit_signal_sessions(self):
        market = self.market()
        daily_targets = {
            self.dates[1]: .5,
            self.dates[2]: .8,
            self.dates[3]: .9,
        }

        def targets_at_signal(*args, **kwargs):
            date = args[4].index[-1]
            return {"SH515880": daily_targets[date]}

        report, trades, _, _, issues, order_log = self.run_engine(
            market, targets_at_signal, signals=self.dates[1:4]
        )
        self.assertEqual(list(trades.signal_date), [
            str(self.dates[1].date()),
            str(self.dates[2].date()),
            str(self.dates[3].date()),
        ])
        self.assertEqual(list(trades.date), [
            str(self.dates[2].date()),
            str(self.dates[3].date()),
            str(self.dates[4].date()),
        ])
        self.assertTrue(trades.quantity.mod(100).eq(0).all())
        self.assertGreater(len(report), 0)
        self.assertEqual(issues["unexecuted_at_end"], 0)
        self.assertTrue(order_log.status.eq("filled").all())

    def test_t_plus_2_freezes_quantity_adv_and_reference_price(self):
        opens = [10., 10., 10., 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
        closes = [10., 10., 10.4, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
        market = self.market(opens=opens, closes=closes)
        market.eligibility.loc[:, "avg_amount"] = 1e7

        _, trades, _, _, _, order_log = self.run_engine(
            market, {"SH515880": .5}, delay=2, signals=self.dates[1:2]
        )
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        row = order_log.iloc[0]
        self.assertEqual(trade.signal_date, str(self.dates[1].date()))
        self.assertEqual(trade.date, str(self.dates[3].date()))
        self.assertAlmostEqual(trade.price, 10.5)
        self.assertEqual(trade.quantity, 5_000.)
        self.assertEqual(row.quantity, 5_000.)
        self.assertEqual(row.adv, 1e7)
        self.assertEqual(row.reference_price, 10.)
        self.assertEqual(row.execution_date, str(self.dates[3].date()))
        self.assertEqual(row.status, "filled")

    def test_t_plus_2_suppresses_repeated_signal_and_cancels_on_new_stop(self):
        market = self.market()
        market.eligibility.loc[:, "avg_amount"] = 1e7

        repeated = lambda *args, **kwargs: {"SH515880": .5}
        _, first_trades, _, _, first_issues, first_orders = self.run_engine(
            market, repeated, delay=2, signals=self.dates[1:4]
        )
        self.assertEqual(len(first_trades), 1)
        self.assertEqual(first_trades.iloc[0].date, str(self.dates[3].date()))
        self.assertEqual(len(first_orders), 1)
        self.assertEqual(first_issues["in_flight_signal_suppressed"], 1)

        def ordinary_change_on_second_signal(
            scores, allowed, current, days, returns, policy, **kwargs
        ):
            date = returns.index[-1]
            return {"SH515880": .8} if date == self.dates[2] else {"SH515880": .5}

        _, preserved_trades, _, _, preserved_issues, preserved_orders = self.run_engine(
            market, ordinary_change_on_second_signal, delay=2,
            signals=self.dates[1:3],
        )
        self.assertEqual(len(preserved_trades), 1)
        self.assertEqual(preserved_trades.iloc[0].date, str(self.dates[3].date()))
        self.assertEqual(preserved_trades.iloc[0].quantity, 5_000.)
        self.assertEqual(len(preserved_orders), 1)
        self.assertEqual(preserved_orders.iloc[0].quantity, 5_000.)
        self.assertEqual(preserved_issues["ordinary_target_changes_preserved"], 1)

        def stop_at_second_signal(scores, allowed, current, days, returns, policy, **kwargs):
            date = returns.index[-1]
            return {"SH515880": 0.} if date == self.dates[2] else {"SH515880": .5}

        _, stopped_trades, _, _, issues, order_log = self.run_engine(
            market, stop_at_second_signal, delay=2, signals=self.dates[1:3]
        )
        self.assertTrue(stopped_trades.empty)
        self.assertEqual(len(order_log), 1)
        self.assertEqual(
            order_log.iloc[0].status, "cancelled_by_stop_or_eligibility"
        )
        self.assertEqual(order_log.iloc[0].cancelled_date, str(self.dates[2].date()))
        self.assertEqual(issues["orders_superseded_by_signal"], 1)

    def test_pending_exit_order_is_never_cancelled_by_a_later_entry_signal(self):
        market = self.market()
        market.eligibility.loc[:, "avg_amount"] = 1e7

        def enter_then_exit(
            scores, allowed, current, days, returns, policy, **kwargs
        ):
            date = returns.index[-1]
            if date < self.dates[3]:
                return {"SH515880": .5}
            if date == self.dates[3]:
                return {}
            return {"SH515880": .5}

        _, trades, _, _, issues, order_log = self.run_engine(
            market, enter_then_exit, delay=2, signals=self.dates[1:5]
        )
        self.assertEqual(list(trades.side), ["buy", "sell"])
        self.assertEqual(list(trades.date), [
            str(self.dates[3].date()),
            str(self.dates[5].date()),
        ])
        self.assertTrue(order_log.status.eq("filled").all())
        self.assertFalse(
            order_log.status.isin(
                ["cancelled_by_stop_or_eligibility", "cancelled_by_new_signal"]
            ).any()
        )
        self.assertEqual(issues["orders_superseded_by_signal"], 0)

    def test_t_plus_2_cancels_pending_order_on_intervening_split(self):
        splits = np.array([1., 1., 2., 2., 2., 2., 2., 2., 2., 2.])
        market = self.market(
            opens=[10.] * len(self.dates),
            closes=[10., 10., 5., 5., 5., 5., 5., 5., 5., 5.],
        )
        market.bars.loc[market.bars.date.ge(self.dates[2]), "adj_close"] = 5.
        action = {
            "event_id": "synthetic-split",
            "type": "split",
            "symbol": "SH515880",
            "effective_date": str(self.dates[2].date()),
            "ratio": 2.,
        }
        _, trades, _, _, issues, order_log = self.run_engine(
            market, {"SH515880": .5}, actions=[action], delay=2,
            signals=self.dates[1:2],
        )
        self.assertTrue(trades.empty)
        self.assertEqual(order_log.iloc[0].status, "cancelled_corporate_action")
        self.assertEqual(order_log.iloc[0].cancelled_date, str(self.dates[2].date()))
        self.assertEqual(issues["corporate_action_order_cancelled"], 1)

    def test_true_shares_cash_dividend_and_split_reconcile(self):
        market = self.market(
            opens=[10.] * len(self.dates),
            closes=[10., 10., 10., 5., 5., 5., 5., 5., 5., 5.],
        )
        dividend = {
            "event_id": "synthetic-dividend",
            "type": "cash_dividend",
            "symbol": "SH515880",
            "record_date": str(self.dates[2].date()),
            "ex_date": str(self.dates[3].date()),
            "payment_date": str(self.dates[5].date()),
            "cash_per_share": .2,
        }
        split = {
            "event_id": "synthetic-split",
            "type": "split",
            "symbol": "SH515880",
            "effective_date": str(self.dates[3].date()),
            "ratio": 2.,
        }
        report, trades, action_log, holdings, _, order_log = self.run_engine(
            market, {"SH515880": .5}, actions=[dividend, split], delay=1,
            signals=self.dates[1:2],
        )
        buy = trades.iloc[0]
        fill_cost = buy.quantity * buy.price + buy.commission
        self.assertEqual(buy.quantity, 5_000.)
        self.assertAlmostEqual(report.loc[self.dates[2], "cash"], 50_000.)
        self.assertAlmostEqual(
            report.loc[self.dates[3], "dividend_receivable"], 1_000.
        )
        split_quantity = holdings[
            holdings.date.eq(self.dates[3])
            & holdings.symbol.eq("SH515880")
        ].quantity.iloc[0]
        self.assertEqual(split_quantity, 10_000.)
        self.assertAlmostEqual(report.loc[self.dates[5], "cash"], 51_000.)
        self.assertEqual(
            list(action_log.type),
            [
                "record_entitlement",
                "split",
                "accrue_dividend",
                "pay_dividend",
            ],
        )
        self.assertEqual(order_log.iloc[0].status, "filled")
        self.assertAlmostEqual(report.loc[self.dates[2], "cash"], 100_000. - fill_cost)

    def test_signal_on_last_session_is_recorded_as_unexecuted(self):
        market = self.market()
        _, trades, _, _, issues, order_log = self.run_engine(
            market, {"SH515880": .5}, delay=2, signals=self.dates[-1:]
        )
        self.assertTrue(trades.empty)
        self.assertEqual(order_log.iloc[0].status, "unexecuted_at_end")
        self.assertEqual(issues["unexecuted_at_end"], 1)
        self.assertIsNone(order_log.iloc[0].cancelled_date)

    def test_weekly_explicit_schedule_exactly_matches_formal_engine(self):
        opens = [10., 10.1, 9.9, 10., 10.2, 10.1, 10., 10., 10., 10.]
        closes = [10., 10., 10.1, 9.9, 10., 10.2, 10., 10., 10., 10.]
        market = self.market(opens=opens, closes=closes)
        start = self.dates[1]
        signals = weekly_sessions(self.dates, start)
        actions = [{
            "event_id": "weekly-parity-dividend",
            "type": "cash_dividend",
            "symbol": "SH515880",
            "record_date": str(self.dates[3].date()),
            "ex_date": str(self.dates[4].date()),
            "payment_date": str(self.dates[8].date()),
            "cash_per_share": .2,
        }]

        def select(*args, **kwargs):
            return {"SH515880": .5}

        expected = simulate_cash(
            market, self.candidate, self.policy, start, actions, selector=select
        )
        actual = simulate_research_cash(
            market, self.candidate, self.policy, start, actions,
            signal_dates=signals, execution_delay_sessions=1, selector=select,
        )
        for expected_frame, actual_frame in zip(expected[:4], actual[:4]):
            pd.testing.assert_frame_equal(actual_frame, expected_frame)
        for key in PRODUCTION_ISSUE_KEYS:
            self.assertEqual(actual[4][key], expected[4][key])

    def test_execution_schedule_is_precomputed_and_validated(self):
        schedule = build_execution_schedule(
            self.dates, self.dates[0], self.dates[1:3], 2
        )
        self.assertEqual(list(schedule.signal_index), [1, 2])
        self.assertEqual(list(schedule.execution_index), [3, 4])
        self.assertEqual(list(schedule.execution_date), [self.dates[3], self.dates[4]])
        with self.assertRaisesRegex(ValueError, "仅支持整数1或2"):
            build_execution_schedule(
                self.dates, self.dates[0], self.dates[1:2], 3
            )
        with self.assertRaisesRegex(ValueError, "非交易日"):
            build_execution_schedule(
                self.dates, self.dates[0], [pd.Timestamp("2026-01-04")], 1
            )


if __name__ == "__main__":
    unittest.main()
