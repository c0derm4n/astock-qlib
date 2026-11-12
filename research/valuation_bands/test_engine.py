"""执行引擎测试: 订单、现金账本、公司行动、部分成交、退出持续与基准。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.valuation_bands import engine as eng
from research.valuation_bands import fixtures
from research.valuation_bands import run as cli
from research.valuation_bands import verify

START = "2016-01-01"
CAL_START = "2011-01-01"
END = "2018-12-31"
PROTOCOL = cli.protocol()
CASE = {"id": "A_pe_weekly", "group": "primary", "strategy": "A", "kind": "primary", "overrides": {}}


def calendar_of(start, end):
    return pd.DatetimeIndex(
        __import__("research.valuation_bands.data", fromlist=["x"]).build_calendar(start, end))


def valuation_path(calendar: pd.DatetimeIndex) -> np.ndarray:
    values = np.full(len(calendar), 15.0)
    for offset, low in enumerate((8.0, 9.0, 10.0, 11.0, 12.0)):
        year = 2011 + offset
        sessions = np.where(calendar.year == year)[0]
        values[sessions[len(sessions) // 3]] = low
    # 2016 上半年低估 -> 分批买入; 下半年高估 -> 清仓; 2017 再次低估; 2018 高估
    values[(calendar >= pd.Timestamp("2016-01-01")) & (calendar < pd.Timestamp("2016-07-01"))] = 9.0
    values[(calendar >= pd.Timestamp("2016-07-01")) & (calendar < pd.Timestamp("2017-01-01"))] = 16.0
    values[(calendar >= pd.Timestamp("2017-01-01")) & (calendar < pd.Timestamp("2017-07-01"))] = 9.0
    values[calendar >= pd.Timestamp("2017-07-01")] = 16.0
    return values


def price_path(calendar: pd.DatetimeIndex) -> np.ndarray:
    values = np.full(len(calendar), 1000.0)
    for offset in range(5):
        year = 2011 + offset
        sessions = np.where(calendar.year == year)[0]
        values[sessions[len(sessions) // 3]] = 900.0
    return values


def pb_path(calendar: pd.DatetimeIndex) -> np.ndarray:
    values = np.full(len(calendar), 7.5)
    for offset, low in enumerate((4.0, 4.5, 5.0, 5.5, 6.0)):
        year = 2011 + offset
        sessions = np.where(calendar.year == year)[0]
        values[sessions[len(sessions) // 3]] = low
    return values


def build_data(close=None, open_=None, amount=None, volume=None, actions=None, start=CAL_START,
               include_b=True):
    calendar = calendar_of(start, END)
    n = len(calendar)
    close = np.full(n, 4.0) if close is None else np.asarray(close, dtype=float)
    open_ = close if open_ is None else np.asarray(open_, dtype=float)
    amount = np.full(n, 2e8) if amount is None else np.asarray(amount, dtype=float)
    volume = np.full(n, 1e7) if volume is None else np.asarray(volume, dtype=float)
    price_map = {"000300": price_path(calendar)}
    valuation_map = {("000300", "PE_TTM"): valuation_path(calendar)}
    etf_map = {"SH510300": {"close": close, "open": open_, "amount": amount, "volume": volume}}
    if include_b:
        price_map["399006"] = price_path(calendar) * 0.5
        valuation_map[("399006", "PB_MRQ")] = pb_path(calendar)
        etf_map["SZ159915"] = {"close": close * 0.5, "open": open_ * 0.5,
                               "amount": amount, "volume": volume}
    return fixtures.make_dataset(
        start, END,
        price_map=price_map,
        valuation_map=valuation_map,
        etf_map=etf_map,
        actions=actions,
    )


def replay(data, end=END, case=None):
    actual_end = data.calendar[data.calendar <= pd.Timestamp(end)][-1]
    return eng.replay(data, PROTOCOL, case or CASE, actual_end, start=pd.Timestamp(START))


class BasicRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_data()
        cls.result = replay(cls.data)

    def test_status_and_daily_cover_every_session(self):
        self.assertEqual(self.result.status, "synthetic_pipeline_validated")
        expected = len(self.data.calendar_range(START, pd.Timestamp(END)))
        self.assertEqual(len(self.result.daily), expected)
        self.assertAlmostEqual(float(self.result.daily["nav"].iloc[0]), 1_000_000.0, places=6)

    def test_first_buy_executes_after_signal_session(self):
        trades = self.result.trades
        first = trades.iloc[0]
        signals = self.result.signals
        first_signal = signals[signals["orders"].astype(str).str.contains("buy")].iloc[0]
        self.assertEqual(first["side"], "buy")
        self.assertGreater(pd.Timestamp(first["execution_date"]), pd.Timestamp(first_signal["session"]))
        self.assertEqual(str(pd.Timestamp(first["execution_date"]).date()), str(first_signal["exec_session"]))

    def test_buy_cost_and_commission(self):
        trade = self.result.trades[self.result.trades["side"] == "buy"].iloc[0]
        notional = float(trade["notional"])
        self.assertAlmostEqual(float(trade["price"]) * float(trade["quantity"]), notional, places=6)
        expected_fee = max(0.20, notional * 0.0003)
        self.assertAlmostEqual(float(trade["commission"]), expected_fee, places=6)
        self.assertEqual(int(trade["quantity"]) % 100, 0)

    def test_batches_are_spaced_by_cooldown(self):
        buys = self.result.trades[self.result.trades["side"] == "buy"]
        positions = [self.data.session_pos(pd.Timestamp(d)) for d in buys["execution_date"]]
        for previous, current in zip(positions, positions[1:]):
            self.assertGreaterEqual(current - previous, 20)

    def test_three_batches_reach_cap_then_exit(self):
        trades = self.result.trades
        year_2016 = trades[pd.to_datetime(trades["execution_date"]).dt.year == 2016]
        buys = year_2016[year_2016["side"] == "buy"]
        sells = year_2016[year_2016["side"] == "sell"]
        self.assertEqual(len(buys), 3)
        self.assertEqual(len(sells), 1)
        self.assertEqual(int(sells.iloc[0]["quantity_after"]), 0)
        daily = self.result.daily
        peak_weight = float(daily["equity_weight"].max())
        self.assertLessEqual(peak_weight, 0.60 + 1e-9)

    def test_exit_resets_round_and_new_round_can_start(self):
        episodes = self.result.episodes
        closed = episodes[episodes["status"] == "closed"]
        self.assertGreaterEqual(len(closed), 1)
        buys_2017 = self.result.trades[
            (pd.to_datetime(self.result.trades["execution_date"]).dt.year == 2017)
            & (self.result.trades["side"] == "buy")
        ]
        self.assertGreaterEqual(len(buys_2017), 1)
        signals = self.result.signals
        self.assertIn("overvaluation_exit", set(signals["reason"]))

    def test_nav_reconciles_with_cash_and_holdings(self):
        daily = self.result.daily
        for row in daily.itertuples(index=False):
            nav = float(row.cash) + float(row.dividend_receivable) + float(row.equity_value)
            self.assertAlmostEqual(float(row.nav), nav, places=6)

    def test_rerun_is_deterministic(self):
        again = replay(build_data())
        self.assertEqual(self.result.metrics, again.metrics)
        pd.testing.assert_frame_equal(self.result.trades, again.trades)
        pd.testing.assert_frame_equal(self.result.daily, again.daily)

    def test_metrics_flags(self):
        metrics = self.result.metrics
        self.assertIn("cagr", metrics)
        self.assertGreaterEqual(metrics["max_drawdown"], 0.0)
        self.assertLessEqual(metrics["max_drawdown"], 1.0)
        self.assertTrue(metrics["goals"]["max_drawdown_pass"])

    def test_calibration_failure_blocks_case_even_with_ledger(self):
        data = build_data()
        valuations = data.frames["index_valuation"]
        mask = (
            (valuations["index_id"] == "000300")
            & (valuations["metric"] == "PE_TTM")
            & (pd.to_datetime(valuations["trade_date"]).dt.year == 2015)
        )
        valuations.loc[mask, "value"] = np.nan
        result = replay(data)
        self.assertEqual(result.status, "data_or_scope_blocked")
        self.assertIn("calibration_data_blocked", result.status_detail)
        self.assertGreater(len(result.daily), 0)


class FillConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = replay(build_data())
        buys = cls.baseline.trades[cls.baseline.trades["side"] == "buy"]
        cls.first_buy = buys.iloc[0]
        cls.first_buy_signal = pd.Timestamp(cls.baseline.signals[
            cls.baseline.signals["session"] == str(
                buys.iloc[0]["signal_date"])
        ].iloc[0]["session"])

    def _data_with(self, **kwargs):
        data = build_data(**kwargs)
        return data

    def test_zero_volume_execution_does_not_increment_steps(self):
        data = build_data()
        signal = self.first_buy_signal
        exec_session = pd.Timestamp(self.first_buy["execution_date"])
        assert data.session_pos(exec_session) - data.session_pos(signal) == 1
        volume = np.full(len(data.calendar), 1e7)
        volume[data.session_pos(exec_session)] = 0.0
        result = replay(build_data(volume=volume))
        orders = result.orders
        target = orders[(orders["signal_date"] == str(signal.date())) & (orders["side"] == "buy")]
        self.assertEqual(str(target.iloc[0]["status"]), "unfilled_no_price_or_suspension")
        row = result.signals[result.signals["session"] == str(signal.date())].iloc[0]
        self.assertEqual(int(row["buy_steps_before"]), 0)
        # 0 成交不增加批次: 后续仍可重新申请第一档
        later = result.signals[result.signals["reason"] == "buy_step_1_requested"]
        self.assertGreaterEqual(len(later), 2)
        first_filled = result.trades[result.trades["side"] == "buy"].iloc[0]
        self.assertGreater(pd.Timestamp(first_filled["execution_date"]), exec_session)

    def test_participation_cap_partial_fill(self):
        data = build_data()
        signal = self.first_buy_signal
        signal_pos = data.session_pos(signal)
        amount = np.full(len(data.calendar), 2e8)
        amount[signal_pos - 19:signal_pos + 1] = 1e6  # 信号日已知的20日均成交额100万
        result = replay(build_data(amount=amount))
        buy = result.trades[result.trades["side"] == "buy"].iloc[0]
        self.assertEqual(str(buy["status"]), "partial")
        self.assertLessEqual(float(buy["notional"]), 20_000 * 1.01)
        row = result.signals[result.signals["session"] == str(signal.date())].iloc[0]
        self.assertEqual(int(row["buy_steps_after_intent"]), 1)

    def test_partial_sell_keeps_exit_intent_until_zero(self):
        # 全局小成交额: 买入与卖出都只能部分成交
        amount = np.full(len(build_data().calendar), 2e6)
        result = replay(build_data(amount=amount))
        reasons = set(result.signals["reason"])
        self.assertIn("exit_intent_repeat", reasons)
        self.assertTrue((result.trades["status"] == "partial").any())
        final_qty = int(result.daily["quantity_SH510300"].iloc[-1])
        self.assertEqual(final_qty, 0)
        sells = result.trades[result.trades["side"] == "sell"]
        self.assertGreater(len(sells), 1)
        # 只有实际清零才重置批次: 首次卖清后的下一次买入从第1批开始
        buys = result.trades[result.trades["side"] == "buy"]
        steps = [row["reason"] for row in buys.to_dict("records")]
        self.assertIn("buy_step_1", steps)

    def test_unexecuted_at_end(self):
        # 以首个买入申请信号日作为研究终点: 该订单没有下一个可执行交易日
        end = str(self.first_buy_signal.date())
        result = replay(build_data(), end=end)
        orders = result.orders[result.orders["status"] == "unexecuted_at_end"]
        self.assertGreaterEqual(len(orders), 1)
        self.assertTrue((orders["execution_date"] == orders["signal_date"]).all())

    def test_delay_t2_executes_two_sessions_after_signal(self):
        data = build_data()
        signal = self.first_buy_signal
        signal_pos = data.session_pos(signal)
        case = {
            "id": "A_stress_delay_t2", "group": "stress", "strategy": "A", "kind": "primary",
            "overrides": {"delay_sessions": 2},
        }
        end = data.calendar[min(signal_pos + 5, len(data.calendar) - 1)]
        result = replay(data, end=end, case=case)
        row = result.signals[result.signals["session"] == str(signal.date())].iloc[0]
        expected = data.calendar[signal_pos + 2]
        self.assertEqual(row["exec_session"], str(expected.date()))
        order = result.orders[
            (result.orders["signal_date"] == str(signal.date()))
            & (result.orders["side"] == "buy")
        ].iloc[0]
        self.assertEqual(order["execution_date"], str(expected.date()))

    def test_t1_sellable_quantity_excludes_same_day_buys(self):
        portfolio = eng.Portfolio(["SH510300"], 1_000_000.0)
        portfolio.quantity["SH510300"] = 1_000
        portfolio.bought_today["SH510300"] = 600
        self.assertEqual(portfolio.sellable_quantity("SH510300"), 400)
        portfolio.bought_today["SH510300"] = 0
        self.assertEqual(portfolio.sellable_quantity("SH510300"), 1_000)

    def test_cash_floor_buy_limit_includes_commission(self):
        exact_floor_break = eng._max_buy_lots_with_cash_floor(
            cash=51_000.0,
            nav_open=1_000_000.0,
            price=10.0,
            min_cash_share=0.05,
            lot_size=100,
            commission_rate=0.0003,
            min_commission=0.20,
        )
        self.assertEqual(exact_floor_break, 0)
        one_lot_above = eng._max_buy_lots_with_cash_floor(
            cash=51_001.0,
            nav_open=1_000_000.0,
            price=10.0,
            min_cash_share=0.05,
            lot_size=100,
            commission_rate=0.0003,
            min_commission=0.20,
        )
        self.assertEqual(one_lot_above, 100)


class ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_data = build_data()
        cls.baseline_result = replay(cls.baseline_data)
        buys = cls.baseline_result.trades[cls.baseline_result.trades["side"] == "buy"]
        cls.first_buy = buys.iloc[0]

    def test_dividend_receivable_then_cash(self):
        data0 = build_data()
        calendar = data0.calendar
        ex = calendar[calendar >= pd.Timestamp("2016-03-15")][0]
        record = calendar[data0.session_pos(ex) - 1]
        pay = calendar[data0.session_pos(ex) + 5]
        actions = fixtures.make_actions("SH510300", [{
            "type": "dividend", "ex_date": str(ex.date()), "record_date": str(record.date()),
            "payment_date": str(pay.date()), "cash_per_share": 0.1,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        actions_rows = result.actions
        self.assertEqual(len(actions_rows), 1)
        amount = float(actions_rows.iloc[0]["amount"])
        self.assertGreater(amount, 0)
        daily = result.daily.set_index("session")
        ex_row = daily.loc[str(ex.date())]
        self.assertAlmostEqual(float(ex_row["dividend_receivable"]), amount, places=6)
        before_pay = daily.loc[str(calendar[data.session_pos(pay) - 1].date())]
        self.assertAlmostEqual(float(before_pay["dividend_receivable"]), amount, places=6)
        pay_row = daily.loc[str(pay.date())]
        self.assertAlmostEqual(float(pay_row["dividend_receivable"]), 0.0, places=6)
        # 支付日现金增加
        prev_row = daily.loc[str(calendar[data.session_pos(pay) - 1].date())]
        self.assertGreater(float(pay_row["cash"]) - float(prev_row["cash"]), amount - 1e-6)
        self.assertEqual(str(actions_rows.iloc[0]["paid_session"]), str(pay.date()))
        self.assertEqual(result.status, "synthetic_pipeline_validated")

    def test_dividend_not_paid_before_ex_date(self):
        data0 = build_data()
        calendar = data0.calendar
        ex = calendar[calendar >= pd.Timestamp("2017-03-15")][0]
        actions = fixtures.make_actions("SH510300", [{
            "type": "dividend", "ex_date": str(ex.date()),
            "record_date": str(calendar[data0.session_pos(ex) - 1].date()),
            "payment_date": str(calendar[data0.session_pos(ex) + 5].date()), "cash_per_share": 0.2,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        daily = result.daily.set_index("session")
        before = daily.loc[str(calendar[data0.session_pos(ex) - 1].date())]
        self.assertAlmostEqual(float(before["dividend_receivable"]), 0.0, places=6)

    def test_split_changes_quantity_not_value(self):
        data0 = build_data()
        calendar = data0.calendar
        ex = calendar[calendar >= pd.Timestamp("2016-03-01")][0]
        close = np.full(len(calendar), 4.0)
        close[data0.session_pos(ex):] = 2.0
        actions = fixtures.make_actions("SH510300", [{
            "type": "split", "ex_date": str(ex.date()), "share_factor": 2.0,
        }])
        data = build_data(close=close, actions=actions)
        result = replay(data)
        daily = result.daily.set_index("session")
        position = data.session_pos(ex)
        before = daily.loc[str(calendar[position - 1].date())]
        after = daily.loc[str(ex.date())]
        self.assertAlmostEqual(float(after["quantity_SH510300"]), float(before["quantity_SH510300"]) * 2, places=6)
        self.assertAlmostEqual(float(after["equity_value"]), float(before["equity_value"]), places=2)
        self.assertAlmostEqual(float(after["cost_basis"]), float(before["cost_basis"]), places=6)

    def test_late_dividend_belongs_to_closed_episode(self):
        actions = fixtures.make_actions("SH510300", [{
            "type": "dividend", "record_date": "2016-07-01", "ex_date": "2016-07-04",
            "payment_date": "2016-07-20", "cash_per_share": 0.10,
        }])
        result = replay(build_data(actions=actions))
        first = result.episodes[result.episodes.status.eq("closed")].iloc[0]
        earned = float(result.actions[result.actions.type.eq("dividend")].amount.sum())
        self.assertGreater(earned, 0)
        self.assertLess(pd.Timestamp(first.exit_session), pd.Timestamp("2016-07-20"))
        self.assertAlmostEqual(first.dividends, earned, places=6)
        self.assertAlmostEqual(first.pnl, first.sell_notional - first.buy_notional + earned - first.fees, places=6)

    def test_execution_day_company_action_cancels_stale_order(self):
        calendar = self.baseline_data.calendar
        signal = pd.Timestamp(self.first_buy["signal_date"])
        exec_session = pd.Timestamp(self.first_buy["execution_date"])
        self.assertEqual(self.baseline_data.session_pos(exec_session), self.baseline_data.session_pos(signal) + 1)
        actions = fixtures.make_actions("SH510300", [{
            "type": "split", "ex_date": str(exec_session.date()), "share_factor": 2.0,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        target = result.orders[
            (result.orders["signal_date"] == str(signal.date()))
            & (result.orders["side"] == "buy")
        ].iloc[0]
        self.assertEqual(target["status"], "cancelled_corporate_action")
        self.assertEqual(int(target["filled_quantity"]), 0)
        later = result.signals[result.signals["reason"] == "buy_step_1_requested"]
        self.assertGreaterEqual(len(later), 2)

    def test_full_risk_sale_preserves_batch_count_and_cooldown(self):
        calendar = self.baseline_data.calendar
        execution = pd.Timestamp(self.first_buy["execution_date"])
        ex = calendar[self.baseline_data.session_pos(execution) + 2]
        payment = calendar[self.baseline_data.session_pos(ex) + 50]
        actions = fixtures.make_actions("SH510300", [{
            "type": "dividend", "ex_date": str(ex.date()),
            "record_date": str(calendar[self.baseline_data.session_pos(ex) - 1].date()),
            "payment_date": str(payment.date()),
            "cash_per_share": 500.0,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        risk_sale = result.trades[
            (result.trades["side"] == "sell")
            & (result.trades["reason"] == "risk_cash_floor")
        ].iloc[0]
        self.assertEqual(int(risk_sale["quantity_after"]), 0)
        next_week = pd.Timestamp(risk_sale["execution_date"]) + pd.Timedelta(days=7)
        following = result.signals[pd.to_datetime(result.signals["session"]) >= next_week].iloc[0]
        self.assertEqual(following["reason"], "hold_cooldown")
        self.assertEqual(int(following["buy_steps_before"]), 1)


class BenchmarkTests(unittest.TestCase):
    def test_missing_benchmark_close_carries_price_instead_of_zero(self):
        data = build_data()
        missing = pd.Timestamp('2016-03-02')
        f = data.frames['etf_bars']
        data.frames['etf_bars'] = f[~(f.symbol.eq('SH510300') & f.trade_date.eq(missing))]
        result = eng.run_benchmark(data, PROTOCOL,
                                  {'id': 'A_etf_buy_hold_cap', 'kind': 'buy_hold', 'strategy': 'A',
                                   'initial_weight': .6}, pd.Timestamp(END), pd.Timestamp(START))
        daily = result['daily'].set_index('session')
        self.assertEqual(daily.loc[missing, 'price_SH510300'], 4.0)
        self.assertEqual(daily.loc[missing, 'missing_data'], 1)
        self.assertAlmostEqual(daily.loc[missing, 'nav'], daily.loc['2016-03-01', 'nav'], places=6)

    def test_cash_benchmark_flat(self):
        data = build_data()
        result = eng.run_benchmark(data, PROTOCOL, {"id": "cash", "kind": "cash"},
                                   pd.Timestamp(END), pd.Timestamp(START))
        self.assertTrue((result["daily"]["nav"] == 1_000_000.0).all())
        self.assertAlmostEqual(result["metrics"]["cagr"], 0.0, places=12)

    def test_buy_hold_enters_next_session_and_holds(self):
        data = build_data()
        result = eng.run_benchmark(data, PROTOCOL,
                                   {"id": "A_etf_buy_hold_95", "kind": "buy_hold", "strategy": "A",
                                    "initial_weight": 0.95},
                                   pd.Timestamp(END), pd.Timestamp(START))
        daily = result["daily"]
        self.assertEqual(float(daily["equity_weight"].iloc[0]), 0.0)
        self.assertGreater(float(daily["equity_weight"].iloc[1]), 0.5)
        self.assertEqual(len(result["trades"]), 1)
        self.assertAlmostEqual(float(daily["quantity_SH510300"].iloc[-1]),
                               float(daily["quantity_SH510300"].iloc[1]), places=6)


class IndependentVerifyTests(unittest.TestCase):
    def test_unused_asset_actions_are_not_account_events(self):
        actions = fixtures.make_actions("SZ159915", [{
            "type": "dividend", "record_date": "2016-03-01", "ex_date": "2016-03-02",
            "payment_date": "2016-03-08", "cash_per_share": .1,
        }])
        data = build_data(actions=actions)
        checked = self._verify(replay(data), data)
        self.assertEqual(checked["status"], "passed")

    @staticmethod
    def _verify(result, data, tamper_amount=0.0):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            result.daily.to_csv(case_dir / "daily_account.csv", index=False)
            result.trades.to_csv(case_dir / "trades.csv", index=False)
            result.actions.to_csv(case_dir / "actions.csv", index=False)
            result.orders.to_csv(case_dir / "orders.csv", index=False)
            (case_dir / "metrics.json").write_text(
                json.dumps(result.metrics, ensure_ascii=False),
                encoding="utf-8",
            )
            if tamper_amount:
                action_path = case_dir / "actions.csv"
                tampered = pd.read_csv(action_path)
                tampered["amount"] = tampered["amount"].astype(float) + tamper_amount
                tampered.to_csv(action_path, index=False)
            symbols = ("SH510300", "SZ159915", "SH512890")
            bars = {symbol: data.etf_bars(symbol) for symbol in symbols}
            sources = {symbol: data.etf_actions(symbol) for symbol in symbols}
            return verify._recompute_case(
                case_dir,
                bars,
                1_000_000.0,
                action_sources=sources,
            )

    def test_independent_recompute_applies_split(self):
        data0 = build_data()
        calendar = data0.calendar
        ex = calendar[calendar >= pd.Timestamp("2016-03-01")][0]
        actions = fixtures.make_actions("SH510300", [{
            "type": "split", "ex_date": str(ex.date()), "share_factor": 2.0,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        checked = self._verify(result, data)
        self.assertEqual(checked["status"], "passed")
        self.assertEqual(checked["max_quantity_diff"], 0.0)

    def test_independent_recompute_detects_dividend_amount_tampering(self):
        data0 = build_data()
        calendar = data0.calendar
        ex = calendar[calendar >= pd.Timestamp("2016-03-15")][0]
        actions = fixtures.make_actions("SH510300", [{
            "type": "dividend", "ex_date": str(ex.date()),
            "record_date": str(calendar[data0.session_pos(ex) - 1].date()),
            "payment_date": str(calendar[data0.session_pos(ex) + 5].date()),
            "cash_per_share": 0.1,
        }])
        data = build_data(actions=actions)
        result = replay(data)
        checked = self._verify(result, data, tamper_amount=1.0)
        self.assertEqual(checked["status"], "failed")
        self.assertAlmostEqual(checked["max_action_amount_diff"], 1.0, places=6)

class DividendSwitchTests(unittest.TestCase):
    """B 空仓红利低波模块与创业板优先、未卖清不得买入的切换规则。"""

    CASE = {"id": "B_pb_dividend", "group": "auxiliary", "strategy": "B", "kind": "auxiliary",
            "overrides": {"dividend_module": True}}

    @staticmethod
    def build(div_volume_zero_on=None):
        calendar = calendar_of(CAL_START, END)
        n = len(calendar)
        lower = (calendar >= pd.Timestamp("2016-07-01")) & (calendar < pd.Timestamp("2017-01-01"))
        pb = pb_path(calendar)
        pb[lower] = 4.5
        pb[calendar >= pd.Timestamp("2017-01-01")] = 7.0
        dy_values = np.full(n, 0.06)
        dy = pd.DataFrame({
            "index_id": "H30269", "trade_date": calendar, "yield_ttm": dy_values,
            "source_id": "fixture", "aggregation_method": "fixture",
        })
        dy["effective_available_at"] = fixtures.next_session_19(calendar, calendar)
        dy["availability_basis"] = "assumed_next_session_19"
        volume_div = np.full(n, 1e7)
        if div_volume_zero_on is not None:
            volume_div[calendar.get_loc(pd.Timestamp(div_volume_zero_on))] = 0.0
        frames_data = fixtures.make_dataset(
            CAL_START, END,
            price_map={"000300": price_path(calendar), "399006": price_path(calendar) * 0.5},
            valuation_map={("000300", "PE_TTM"): valuation_path(calendar), ("399006", "PB_MRQ"): pb},
            etf_map={
                "SH510300": {"close": np.full(n, 4.0), "open": np.full(n, 4.0)},
                "SZ159915": {"close": np.full(n, 2.0), "open": np.full(n, 2.0)},
                "SH512890": {"close": np.full(n, 1.0), "open": np.full(n, 1.0),
                             "amount": np.full(n, 2e8), "volume": volume_div},
            },
            dividend_yield=dy,
        )
        return frames_data

    def test_switch_sells_dividend_before_buying_chinext(self):
        data = self.build()
        result = eng.replay(data, PROTOCOL, self.CASE, pd.Timestamp(END), start=pd.Timestamp(START))
        trades = result.trades
        div_buys = trades[(trades["side"] == "buy") & (trades["symbol"] == "SH512890")]
        self.assertGreaterEqual(len(div_buys), 1)
        b_buys = trades[(trades["side"] == "buy") & (trades["symbol"] == "SZ159915")]
        self.assertGreaterEqual(len(b_buys), 1)
        first_b_signal = str(b_buys.iloc[0]["signal_date"])
        window_sells = trades[(trades["side"] == "sell") & (trades["symbol"] == "SH512890")
                              & (trades["signal_date"] == first_b_signal)]
        self.assertEqual(len(window_sells), 1)
        switch_session = str(b_buys.iloc[0]["execution_date"])
        self.assertEqual(str(window_sells.iloc[0]["execution_date"]), switch_session)
        # 先卖后买: 在成交顺序中该卖出出现在该买入之前
        first_buy_index = trades.index[(trades["symbol"] == "SZ159915")].min()
        sell_index = window_sells.index[0]
        self.assertLess(sell_index, first_buy_index)
        signal_row = result.signals[result.signals["session"] == first_b_signal].iloc[0]
        self.assertIn("sell:SH512890:dividend_switch_sell", signal_row["orders"])
        self.assertIn("buy:SZ159915", signal_row["orders"])
        self.assertEqual(str(result.status), "synthetic_pipeline_validated")

    def test_buy_chinext_cancelled_when_dividend_not_fully_sold(self):
        baseline = self.build()
        first = eng.replay(baseline, PROTOCOL, self.CASE, pd.Timestamp(END), start=pd.Timestamp(START))
        b_buys = first.trades[(first.trades["side"] == "buy") & (first.trades["symbol"] == "SZ159915")]
        exec_session = pd.Timestamp(b_buys.iloc[0]["execution_date"])
        data = self.build(div_volume_zero_on=exec_session)
        result = eng.replay(data, PROTOCOL, self.CASE, pd.Timestamp(END), start=pd.Timestamp(START))
        orders = result.orders
        switch_sell = orders[(orders["symbol"] == "SH512890") & (orders["side"] == "sell")
                            & (orders["execution_date"] == str(exec_session.date()))]
        self.assertEqual(len(switch_sell), 1)
        self.assertEqual(str(switch_sell.iloc[0]["status"]), "unfilled_no_price_or_suspension")
        cancelled = orders[(orders["symbol"] == "SZ159915") & (orders["side"] == "buy")
                           & (orders["status"] == "cancelled_dependency_not_flat")]
        self.assertGreaterEqual(len(cancelled), 1)
        locked = orders[(orders["symbol"] == "SH512890")
                        & (orders["reason"].isin(
                            ("dividend_exit_intent_repeat", "dividend_switch_sell")
                        ))
                        & (orders["signal_date"] > str(exec_session.date()))]
        self.assertGreaterEqual(len(locked), 1)
        # 红利未卖清之前不得买入创业板
        buys_before = result.trades[(result.trades["symbol"] == "SZ159915")
                                   & (pd.to_datetime(result.trades["execution_date"]) <= exec_session)]
        self.assertEqual(len(buys_before), 0)


if __name__ == "__main__":
    unittest.main()
