import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import config
from backend.artifacts import write_json_atomic
from src.account import initialize, apply_fills, apply_actions
from src.forward import freeze_signal, record_close, forward_evidence, observation_dir
from src.models import learn_bundle, predict_bundle, fit_production
from src.research import evaluate_gate, stress_scenarios
from src.runs import publish, load_production, artifact_path, read_protocol


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = patch.object(config, "OUTPUT_DIR", self.root)
        p.start()
        self.addCleanup(p.stop)
        self.policy = read_protocol()

    def test_protocol_drives_all_stress_scenarios_and_losses_block(self):
        policy = dict(self.policy, stress_fill_fractions=[1., .5, .25], stress_execution_delays=[1, 2, 3])
        scenarios = dict(stress_scenarios(policy))
        self.assertEqual(scenarios["fill_0.25"], {"fill_fraction": .25})
        self.assertEqual(scenarios["delay_2_days"], {"delay": 3})
        metrics = {"start": "2021-01-04", "end": "2026-07-27", "annual_returns": {"2024": .1},
                   "cumulative_return": .1, "max_drawdown": -.1}
        stress = {"slippage_2": metrics, "partial_fills": dict(metrics, cumulative_return=-.01)}
        result = evaluate_gate(metrics, stress, {"ci_low": .01}, policy, 126)
        self.assertFalse(result["checks"]["execution_stress_survives"])

    def test_model_manifest_and_model_bytes_are_verified(self):
        directory = self.root / "runs" / "test"
        directory.mkdir(parents=True)
        bundle = {"run_id": "test", "protocol_hash": "p", "code_fingerprint": "c"}
        model = directory / "production_model.pkl"
        model.write_bytes(pickle.dumps(bundle))
        publish("test", directory, {"run_id": "test", "protocol_hash": "p", "code": {"fingerprint": "c"}})
        self.assertEqual(load_production()[0], bundle)
        self.assertEqual(artifact_path("predictions.pkl"), (directory / "predictions.pkl").resolve())
        model.write_bytes(pickle.dumps(dict(bundle, run_id="other")))
        with self.assertRaisesRegex(ValueError, "模型内容"):
            load_production()
        write_json_atomic(self.root / "current_run.json", {"run_id": "../../elsewhere"})
        with self.assertRaisesRegex(ValueError, "指针损坏"):
            artifact_path("predictions.pkl")

    def test_saved_preprocessor_is_reused_without_refit(self):
        x = pd.DataFrame({"a": [1., 2., np.nan, 4.], "b": [2., 3., 4., 5.]})
        bundle = learn_bundle(x, pd.Series([0., .1, .2, .3]), "ridge", 1)
        restored = pickle.loads(pickle.dumps(bundle))
        center = restored["preprocessor"].named_steps["scale"].center_.copy()
        future = pd.DataFrame({"a": [1000., np.nan], "b": [4000., 5000.]})
        pd.testing.assert_series_equal(predict_bundle(bundle, future), predict_bundle(restored, future))
        np.testing.assert_array_equal(center, restored["preprocessor"].named_steps["scale"].center_)
        with self.assertRaises(ValueError):
            predict_bundle(restored, future[["b", "a"]])

    def test_production_refit_only_uses_mature_labels(self):
        calendar = pd.bdate_range("2022-01-03", periods=650)
        index = pd.MultiIndex.from_product([calendar, ["SH510300", "SH511260"]], names=["datetime", "instrument"])
        x = pd.DataFrame({"a": np.arange(len(index))}, index=index)
        y = pd.Series(np.tile([.1, .2], len(calendar)), index=index)
        ends = pd.Series(index.get_level_values("datetime") + pd.offsets.BDay(21), index=index)
        market = SimpleNamespace(calendar=calendar, label_ends=ends, fingerprint="test")
        calls = []
        def learn(features, labels, *args, **kwargs):
            calls.append(features.index)
            return {"best_iteration": 3}
        with patch("src.models.sample_frame", return_value=(x, y)), patch("src.models.learn_bundle", side_effect=learn):
            bundle = fit_production(market, self.policy)
        self.assertLessEqual(pd.Timestamp(bundle["train_label_end"]), calendar[-1])
        self.assertEqual(pd.Timestamp(bundle["train_signal_end"]), calendar[-22])
        self.assertTrue((ends.reindex(calls[-1]) <= calendar[-1]).all())
        self.assertTrue((ends.reindex(calls[0]) < calendar[-1] - pd.DateOffset(years=1)).all())

    def test_corporate_actions_are_atomic_idempotent_and_invalidate_orders(self):
        path = self.root / "account.json"
        initialize(path, 10000, {"SH510300": {"quantity": 100, "average_cost": 4., "buy_date": "2026-07-01"}})
        split = {"event_id": "split1", "source": "test receipt", "type": "split", "symbol": "SH510300",
                 "effective_date": "2026-07-20", "ratio": 2}
        dividend = dict(split, event_id="cash1", type="cash_dividend", cash_amount=10)
        state = apply_actions(path, [split, dividend])
        self.assertEqual(state["holdings"]["SH510300"]["quantity"], 200)
        self.assertEqual(state["holdings"]["SH510300"]["average_cost"], 2.)
        self.assertEqual(state["cash"], 10010)
        self.assertIsNone(state["active_decision_id"])
        self.assertEqual(state, apply_actions(path, [split, dividend]))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            apply_actions(path, [dict(dividend, event_id="cash2"), dict(split, ratio=3)])
        self.assertEqual(path.read_bytes(), before)
        settled = apply_actions(path, [dict(dividend, event_id="settle1", type="delist_settlement", cash_amount=400)])
        self.assertEqual(settled["cash"], 10410)
        self.assertEqual(settled["holdings"], {})

    def test_existing_holding_topup_cannot_bypass_t_plus_one(self):
        path = self.root / "account.json"
        initialize(path, 10000, {"SH510300": {"quantity": 100, "average_cost": 4., "buy_date": "2026-07-01"}})
        decision = {"decision_id": "d", "account_revision": 0, "execution_date": "2026-07-27", "orders": [
            {"order_id": "buy", "symbol": "SH510300", "side": "buy", "quantity": 100},
            {"order_id": "sell", "symbol": "SH510300", "side": "sell", "quantity": 200}]}
        fill = {"execution_id": "1", "decision_id": "d", "order_id": "buy", "symbol": "SH510300",
                "side": "buy", "quantity": 100, "price": 4., "fee": .2, "executed_at": "2026-07-27T09:30:00+08:00"}
        apply_fills(path, decision, [fill])
        with self.assertRaisesRegex(ValueError, "T\\+1"):
            apply_fills(path, decision, [dict(fill, execution_id="2", order_id="sell", side="sell", quantity=200)])
        sold = apply_fills(path, decision, [dict(fill, execution_id="3", order_id="sell", side="sell")])
        self.assertEqual(sold["holdings"]["SH510300"]["quantity"], 100)

    def test_forward_rejects_backfill_replacement_and_unfrozen_close(self):
        manifest = {"protocol_hash": "p", "code": {"fingerprint": "c"}, "protocol": self.policy}
        decision = {"decision_id": "d", "execution_date": "2026-07-27", "readiness": {"data_age_calendar_days": 0}}
        with patch("src.forward._now", return_value=pd.Timestamp("2026-07-27T19:00:00+08:00")):
            self.assertFalse(freeze_signal(decision, manifest))
            with patch("src.forward.load_production", return_value=({}, manifest)):
                with self.assertRaisesRegex(ValueError, "未冻结"):
                    record_close()
        with patch("src.forward._now", return_value=pd.Timestamp("2026-07-24T19:00:00+08:00")):
            self.assertTrue(freeze_signal(decision, manifest))
            self.assertTrue(freeze_signal(decision, manifest))
            with self.assertRaisesRegex(ValueError, "禁止事后替换"):
                freeze_signal(dict(decision, decision_id="other"), manifest)

    def test_forward_missing_sessions_or_reset_accounts_block(self):
        directory = observation_dir("p", "c")
        def add(day, opening_nav=100):
            write_json_atomic(directory / "signals" / f"{day}.json", {
                "frozen_at": f"{day}T08:00:00+08:00", "decision": {"decision_id": day, "execution_date": day}})
            write_json_atomic(directory / "closes" / f"{day}.json", {"date": day, "recorded_at": f"{day}T19:00:00+08:00",
                "decision_id": day, "nav": 100., "opening_reference_nav": opening_nav, "execution_aligned": True})
        add("2026-07-27")
        add("2026-07-29")
        self.assertFalse(forward_evidence("p", "c")["continuous"])
        add("2026-07-28", 101.)
        self.assertFalse(forward_evidence("p", "c")["account_reconciled"])
        add("2026-07-28")
        self.assertTrue(forward_evidence("p", "c")["execution_aligned"])

    def test_unfilled_orders_cannot_count_as_aligned_forward_execution(self):
        path = self.root / "account.json"
        initialize(path, 10000)
        manifest = {"protocol_hash": "p", "code": {"fingerprint": "c"}, "protocol": self.policy}
        decision = {"decision_id": "d", "execution_date": "2026-07-27", "account_nav": 10000,
                    "orders": [{"order_id": "o", "quantity": 100}], "readiness": {"data_age_calendar_days": 0}}
        with patch("src.forward._now", return_value=pd.Timestamp("2026-07-24T19:00:00+08:00")):
            freeze_signal(decision, manifest, path)
        market = SimpleNamespace(calendar=pd.DatetimeIndex(["2026-07-27"]), fingerprint="data",
                                 wide=lambda _: pd.DataFrame({"SH510300": [4.]}))
        with patch("src.forward._now", return_value=pd.Timestamp("2026-07-27T19:00:00+08:00")), \
             patch("src.forward.load_production", return_value=({}, manifest)), \
             patch("src.forward.load_market", return_value=market):
            receipt = json.loads(record_close(path).read_text(encoding="utf-8"))
        self.assertEqual(receipt["minimum_order_fill_ratio"], 0)
        self.assertFalse(forward_evidence("p", "c", path)["execution_aligned"])


if __name__ == "__main__":
    unittest.main()
