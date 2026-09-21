import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import config
from src.market import Market, causal_features
from src.portfolio import target_weights, plan_orders, constrain_weights
from src.simulation import simulate, block_bootstrap_mean
from src.account import initialize, read_account, apply_fills
from src.decide import validate_readiness, next_session


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((config.BASE_DIR / "research_protocol.json").read_text(encoding="utf-8"))
        self.symbols = ["SH515070", "SZ159995", "SH515880", "SH510300", "SH511260", "SH518880"]
        self.returns = pd.DataFrame(np.random.default_rng(1).normal(0, .01, (100, 6)), columns=self.symbols)

    def test_concentration_and_risk_caps(self):
        weights = constrain_weights(dict.fromkeys(self.symbols, .4), self.returns, self.policy)
        self.assertLessEqual(weights.max(), .2)
        self.assertLessEqual(weights.sum(), .95)
        self.assertLessEqual(weights.reindex(self.symbols[:3]).sum(), .35 + 1e-10)
        annual_vol = np.sqrt(weights.to_numpy() @ self.returns.cov().to_numpy() @ weights.to_numpy() * 252)
        self.assertLessEqual(annual_vol, .15 + 1e-10)

    def test_initial_turnover_is_limited(self):
        scores = pd.Series(range(6), index=self.symbols)
        targets = target_weights(scores, self.symbols, {}, {}, self.returns, self.policy)
        self.assertLessEqual(sum(targets.values()), .2 + 1e-10)

    def test_protected_holding_does_not_create_extra_position(self):
        current = {s: .1 for s in self.symbols[:2]}
        policy = dict(self.policy, topk=2, max_group_weight=1., max_equity_weight=1.)
        scores = pd.Series([0., .1, 1.], index=self.symbols[:3])
        targets = target_weights(scores, self.symbols[:3], current, {s: 1 for s in current}, self.returns, policy)
        self.assertNotIn(self.symbols[2], targets)

    def test_ineligible_holdings_exit_even_during_protection(self):
        targets = target_weights(pd.Series({"SH510300": 1.}), ["SH510300"], {"SH515070": .2},
                                 {"SH515070": 1}, self.returns, self.policy)
        self.assertNotIn("SH515070", targets)

    def test_order_capacity_lots_and_unknown_valuation(self):
        orders = plan_orders({"SH510300": .2}, {}, 1000000, {"SH510300": 4}, {"SH510300": 1000000}, self.policy)
        self.assertEqual(orders[0]["quantity"] % 100, 0)
        self.assertLessEqual(orders[0]["quantity"] * 4, 20000)
        with self.assertRaises(ValueError):
            plan_orders({}, {"SH510300": 100}, 1000, {}, {}, self.policy)

    def test_features_do_not_depend_on_future_prices(self):
        days = pd.bdate_range("2024-01-01", periods=200)
        bars = pd.DataFrame({"date": days, "symbol": "SH510300", "adj_open": np.arange(200)+100.,
                             "adj_close": np.arange(200)+101., "close": np.arange(200)+101.,
                             "high": np.arange(200)+102., "low": np.arange(200)+99.,
                             "amount": 1e8, "volume": 100000.})
        old, label, ends = causal_features(bars)
        bars.loc[150:, ["adj_open", "adj_close", "close", "high", "low"]] *= 2
        new, _, _ = causal_features(bars)
        pd.testing.assert_frame_equal(old.iloc[:150], new.iloc[:150])
        self.assertEqual(ends.iloc[0], days[21])
        self.assertAlmostEqual(label.iloc[0], (121 / 101 - 1))

    def test_block_bootstrap_is_reproducible(self):
        values = np.random.default_rng(1).normal(0, .01, 300)
        self.assertEqual(block_bootstrap_mean(values), block_bootstrap_mean(values))

    def test_simulation_executes_only_after_signal_and_conserves_account(self):
        dates = pd.bdate_range("2025-01-01", periods=160)
        rows = []
        for symbol in ("SH510300", "SH511260"):
            for i, day in enumerate(dates):
                close = 4 + .001*i
                rows.append({"date": day, "symbol": symbol, "open": close, "close": close,
                             "high": close+.01, "low": close-.01, "adj_open": close, "adj_close": close,
                             "adj_factor": 1., "volume": 1e7, "amount": 1e9})
        bars = pd.DataFrame(rows)
        feature, labels, ends = causal_features(bars)
        eligibility = bars[["date", "symbol"]].assign(eligible=True, avg_amount=1e9)
        market = Market(bars, eligibility, feature, labels, ends, dates, "test", {})
        index = pd.MultiIndex.from_product([dates[125:], ["SH510300", "SH511260"]], names=["datetime", "instrument"])
        scores = pd.Series([1., .1] * 35, index=index)
        report, trades, issues = simulate(market, scores, self.policy)
        self.assertGreater(len(trades), 0)
        self.assertTrue((pd.to_datetime(trades["date"]) > pd.to_datetime(trades["signal_date"])).all())
        self.assertTrue((report["cash"] >= 0).all())
        self.assertTrue((trades["participation"] <= self.policy["max_adv_participation"] + 1e-9).all())
        nav = (1 + report["return"] - report["cost"]).cumprod() * self.policy["initial_capital"]
        np.testing.assert_allclose(nav, report["account"], atol=1e-7)
        changed = bars.copy()
        changed.loc[changed["date"] >= dates[150], ["open", "close", "adj_open", "adj_close"]] *= 1.01
        market.bars = changed
        revised, _, _ = simulate(market, scores, self.policy)
        pd.testing.assert_frame_equal(report.loc[:dates[149]], revised.loc[:dates[149]])

    def test_exchange_calendar_and_readiness_fail_closed(self):
        self.assertEqual(next_session("2026-09-30"), "2026-10-08")
        bundle = {"train_label_end": "2026-07-24", "trained_as_of": "2026-07-27"}
        manifest = {"investment_gate": {"approved": False}}
        with self.assertRaisesRegex(ValueError, "行情已过期"):
            validate_readiness(bundle, manifest, "2026-07-27", now="2026-09-12T19:00:00+08:00")
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            validate_readiness(bundle, manifest, "2026-07-27", now="2026-07-27T14:30:00+08:00", paper=True)
        with self.assertRaisesRegex(ValueError, "标签"):
            validate_readiness(bundle, manifest, "2026-07-20", now="2026-07-27T19:00:00+08:00", paper=True)


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "account.json"
        initialize(self.path, 10000)
        self.decision = {"decision_id": "d1", "account_revision": 0, "execution_date": "2026-07-27", "orders": [
            {"order_id": "o1", "symbol": "SH510300", "side": "buy", "quantity": 200}]}
        self.fill = {"execution_id": "e1", "decision_id": "d1", "order_id": "o1", "symbol": "SH510300",
                     "side": "buy", "quantity": 100, "price": 4., "fee": 1., "executed_at": "2026-07-27T09:30:00+08:00"}

    def test_missing_corrupt_and_legacy_accounts_fail_closed(self):
        for value in ("{", '{"positions":[]}'):
            self.path.write_text(value)
            with self.assertRaises(ValueError):
                read_account(self.path)
        with self.assertRaises(ValueError):
            read_account(Path(self.tmp.name) / "missing.json")

    def test_duplicate_fills_are_idempotent_and_partial_fills_accumulate(self):
        first = apply_fills(self.path, self.decision, [self.fill])
        self.assertEqual(first["cash"], 9599)
        duplicate = apply_fills(self.path, self.decision, [self.fill])
        self.assertEqual(first, duplicate)
        second = apply_fills(self.path, self.decision, [dict(self.fill, execution_id="e2")])
        self.assertEqual(second["holdings"]["SH510300"]["quantity"], 200)
        self.assertEqual(second["cash"], 9198)

    def test_conflicting_duplicate_overfill_and_negative_cash_are_rejected(self):
        apply_fills(self.path, self.decision, [self.fill])
        before = self.path.read_bytes()
        for changed in (dict(self.fill, price=5), dict(self.fill, execution_id="e2", quantity=300),
                        dict(self.fill, execution_id="e2", price=1000)):
            with self.assertRaises(ValueError):
                apply_fills(self.path, self.decision, [changed])
            self.assertEqual(self.path.read_bytes(), before)

    def test_batch_is_atomic(self):
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            apply_fills(self.path, self.decision, [self.fill, dict(self.fill, execution_id="bad", side="sell")])
        self.assertEqual(self.path.read_bytes(), before)

    def test_initialize_never_overwrites(self):
        with self.assertRaises(ValueError):
            initialize(self.path, 5000)

    def test_future_and_wrong_day_fills_rejected(self):
        for timestamp in ("2030-07-27T09:30:00+08:00", "2026-07-28T09:30:00+08:00", "2026-07-27T09:30:00"):
            with self.assertRaises(ValueError):
                apply_fills(self.path, self.decision, [dict(self.fill, executed_at=timestamp)])


if __name__ == "__main__":
    unittest.main()
