"""CLI 与运行框架测试: 登记冻结、哈希守卫、不可覆盖、缺口阻塞与独立对账。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from research.valuation_bands import data as dm
from research.valuation_bands import fixtures
from research.valuation_bands import run as cli
from research.valuation_bands import verify as vf
from research.valuation_bands.test_engine import build_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = [
    PROJECT_ROOT / "universe.py",
    PROJECT_ROOT / "strategy_registry.json",
    PROJECT_ROOT / "research_protocol.json",
    PROJECT_ROOT / "output" / "strategies" / "current.json",
    PROJECT_ROOT / "data" / "formal" / "market.duckdb",
]


class ProtocolTests(unittest.TestCase):
    def test_protocol_has_25_cases_in_expected_groups(self):
        doc = cli.protocol()
        cases = doc["cases"]
        self.assertEqual(len(cases), 25)
        self.assertEqual(len({case["id"] for case in cases}), 25)
        counts = {}
        for case in cases:
            counts[case["group"]] = counts.get(case["group"], 0) + 1
        self.assertEqual(counts, {"primary": 2, "stress": 4, "sensitivity": 16, "auxiliary": 3})
        for case in cases:
            self.assertIn(case["strategy"], ("A", "B"))

    def test_parameter_overrides_are_isolated(self):
        doc = cli.protocol()
        base = {case["id"]: case for case in doc["cases"]}
        default = cli.engine_mod.resolve_case(doc, base["A_pe_weekly"])
        wider = cli.engine_mod.resolve_case(doc, base["A_sens_buy_upper_1_15"])
        closer = cli.engine_mod.resolve_case(doc, base["A_sens_exit_q_0_80"])
        self.assertEqual(wider.band_upper_mult, 1.15)
        self.assertEqual(wider.exit_q, default.exit_q)
        self.assertEqual(wider.w_cap, default.w_cap)
        self.assertEqual(closer.exit_q, 0.80)
        self.assertEqual(closer.band_upper_mult, default.band_upper_mult)
        fixed = cli.engine_mod.resolve_case(doc, base["A_pe_fixed14_34"])
        self.assertEqual(fixed.fixed_exit_level, 14.34)
        gate = cli.engine_mod.resolve_case(doc, base["A_pe_ratio_gate"])
        self.assertEqual(gate.gate, "ratio")
        dividend = cli.engine_mod.resolve_case(doc, base["B_pb_dividend"])
        self.assertTrue(dividend.dividend_module)

    def test_sensitivity_band_keeps_mid_and_low_bands(self):
        doc = cli.protocol()
        case = {c["id"]: c for c in doc["cases"]}["A_sens_buy_upper_1_15"]
        params = cli.engine_mod.calibration_params(doc, cli.engine_mod.resolve_case(doc, case))
        self.assertEqual(params["band_mid_mult"], 1.0)
        self.assertEqual(params["band_low_mult"], 0.9)
        self.assertEqual(params["band_upper_mult"], 1.15)


class RunFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        data = build_data()
        cls.manifest = fixtures.write_snapshot(
            cls.root / "snapshot", data.frames, ("2011-01-01", "2018-12-31"))
        cls.out = cls.root / "output"
        cls.production_hashes = {str(p): (dm.sha256_file(p) if p.exists() else None) for p in PRODUCTION_FILES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.patcher = mock.patch.object(cli, "OUT_DIR", type(self).out)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _register(self, name="run-a"):
        code = cli.main(["register", "--manifest", str(self.manifest), "--directory-name", name])
        self.assertEqual(code, 0)
        return self.out / "runs" / name

    def test_register_freezes_hashes_and_cases(self):
        run_dir = self._register("run-register")
        registration = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
        self.assertEqual(registration["actual_end"], "2018-12-28")  # 2018-12-31 休市
        self.assertFalse(registration["investment_admission"])
        self.assertTrue(registration["screenshot_reviewed"])
        self.assertEqual(len(registration["cases"]), 25)
        self.assertEqual(registration["code_hashes"]["data.py"], dm.sha256_file(cli.PACKAGE_DIR / "data.py"))
        self.assertEqual(registration["protocol_hash"], dm.sha256_file(cli.PROTOCOL_PATH))
        blocked = [c["id"] for c in registration["cases"] if c["status"] == "data_or_scope_blocked"]
        self.assertIn("B_pb_dividend", blocked)       # 缺少 H30269 股息率与 512890 行情
        self.assertIn("A_pe_ratio_gate", blocked)     # 缺少 H00300/885007.WI
        self.assertNotIn("A_pe_weekly", blocked)

    def test_synthetic_guard_requires_flag(self):
        run_dir = self._register("run-guard")
        code = cli.main(["run", "--directory", str(run_dir), "--group", "primary"])
        self.assertEqual(code, 3)

    def test_code_hash_mismatch_refused(self):
        run_dir = self._register("run-hash")
        registration = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
        registration["code_hashes"]["engine.py"] = "0" * 64
        (run_dir / "registration.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        code = cli.main(["run", "--directory", str(run_dir), "--group", "primary", "--allow-synthetic"])
        self.assertEqual(code, 5)

    def test_protocol_hash_mismatch_refused(self):
        run_dir = self._register("run-protocol")
        registration = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
        registration["protocol_hash"] = "f" * 64
        (run_dir / "registration.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        code = cli.main(["run", "--directory", str(run_dir), "--group", "primary", "--allow-synthetic"])
        self.assertEqual(code, 4)

    def test_primary_run_and_verification(self):
        run_dir = self._register("run-primary")
        code = cli.main(["run", "--directory", str(run_dir), "--group", "primary", "--allow-synthetic"])
        self.assertEqual(code, 0)
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(results["cases"]), 2)
        for case in results["cases"]:
            self.assertEqual(case["status"], "synthetic_pipeline_validated")
        report = vf.verify_run(run_dir)
        self.assertEqual(report["failed"], 0)
        self.assertGreaterEqual(report["checked_cases"], 2)
        self.assertTrue((run_dir / "verification.json").exists())

        # 第二次运行不得覆盖终态 case
        orders_path = run_dir / "cases" / "A_pe_weekly" / "orders.csv"
        before = orders_path.read_bytes()
        code = cli.main(["run", "--directory", str(run_dir), "--group", "primary", "--allow-synthetic"])
        self.assertEqual(code, 0)
        self.assertEqual(orders_path.read_bytes(), before)

    def test_report_generated(self):
        run_dir = self._register("run-report")
        cli.main(["run", "--directory", str(run_dir), "--group", "primary", "--allow-synthetic"])
        code = cli.main(["report", "--directory", str(run_dir)])
        self.assertEqual(code, 0)
        text = (run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("估值择时研究报告", text)
        self.assertIn("投资准入", text)
        self.assertIn("A_pe_weekly", text)

    def test_production_files_untouched(self):
        for path, digest in self.production_hashes.items():
            if digest is None:
                continue
            self.assertEqual(dm.sha256_file(Path(path)), digest, f"{path} 被修改")


class GapTests(unittest.TestCase):
    def test_missing_valuation_blocks_primary(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            data = build_data()
            frames = {k: v for k, v in data.frames.items() if k != "index_valuation"}
            manifest = fixtures.write_snapshot(Path(tmp.name) / "snap", frames, ("2011-01-01", "2018-12-31"))
            snapshot = dm.load_snapshot(manifest)
            gaps = dm.check_gaps(snapshot, cli.protocol())
            self.assertFalse(gaps["can_run_primary"])
            self.assertIn("index_valuation:000300:PE_TTM", gaps["blocking_series"])
            self.assertFalse(cli.freeze_actual_end(snapshot, cli.protocol()) is not None)
        finally:
            tmp.cleanup()

    def test_monthly_valuation_flagged_as_frequency_insufficient(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            data = build_data()
            frame = data.frames["index_valuation"].copy()
            months = frame["trade_date"].dt.to_period("M").astype(str)
            frame["trade_date"] = frame["trade_date"].dt.to_period("M").dt.to_timestamp()
            frame = frame.groupby(["index_id", "metric", "trade_date"]).first().reset_index()
            frames = dict(data.frames)
            frames["index_valuation"] = frame
            manifest = fixtures.write_snapshot(Path(tmp.name) / "snap", frames, ("2011-01-01", "2018-12-31"))
            snapshot = dm.load_snapshot(manifest)
            gaps = dm.check_gaps(snapshot, cli.protocol())
            codes = {g["code"] for g in gaps["gaps"] if g["series"] == "index_valuation:000300:PE_TTM"}
            self.assertIn("frequency_insufficient", codes)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
