"""临时产物与 ASGI 请求测试；无需启动服务或连接行情接口。"""
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd

from backend import data_service
from backend.main import app
import config


class ArtifactFixture:
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(config, "OUTPUT_DIR", self.root))
        for name, file in (("_PRED_FILE", "predictions.pkl"), ("_REPORT_FILE", "report.csv"),
                           ("_RUN_META_FILE", "meta.json")):
            self.stack.enter_context(patch.object(data_service, name, self.root / file))
        data_service._cache.clear()
        self.addCleanup(data_service._cache.clear)

    def write_predictions(self, values=None):
        index = pd.MultiIndex.from_product([
            pd.to_datetime(["2026-09-10", "2026-09-11"]), ["SH510300", "SZ159915"]
        ], names=["datetime", "instrument"])
        pred = pd.Series(values if values is not None else [0.1, 0.2, 0.3, 0.4], index=index)
        pred.to_pickle(data_service._PRED_FILE)
        return pred

    def write_report(self, returns, bench=None, cost=None):
        df = pd.DataFrame({
            "return": returns,
            "bench": bench if bench is not None else [0.0] * len(returns),
            "cost": cost if cost is not None else [0.001] * len(returns),
        }, index=pd.date_range("2026-09-01", periods=len(returns)))
        df.to_csv(data_service._REPORT_FILE)
        return df


class DataServiceTests(ArtifactFixture, unittest.TestCase):
    def test_prediction_queries_reuse_loaded_index_and_preserve_ranking(self):
        self.write_predictions()
        with patch.object(pd, "read_pickle", wraps=pd.read_pickle) as read:
            self.assertEqual(data_service.available_dates(), ["2026-09-10", "2026-09-11"])
            result = data_service.get_picks("2026-09-12", 1)
            self.assertEqual(result["date"], "2026-09-11")
            self.assertEqual(result["rows"][0]["code"], "SZ159915")
            data_service.get_picks(None, 2)
        read.assert_called_once()

    def test_predictions_reload_after_file_change(self):
        self.write_predictions()
        first = data_service.get_picks(None, 1)
        self.write_predictions([0.1, 0.2, 0.9, 0.4])
        second = data_service.get_picks(None, 1)
        self.assertNotEqual(first["rows"][0]["code"], second["rows"][0]["code"])

    def test_cache_invalidates_when_size_changes_with_same_mtime(self):
        path = self.root / "sample.txt"
        path.write_text("old", encoding="utf-8")
        stat = path.stat()
        loader = lambda p: p.read_text(encoding="utf-8")
        self.assertEqual(data_service._cached(path, loader), "old")
        path.write_text("new contents", encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(data_service._cached(path, loader), "new contents")

    def test_concurrent_cold_queries_load_once(self):
        path = self.root / "sample.txt"
        path.write_text("cached", encoding="utf-8")
        barrier = threading.Barrier(8)
        loader = Mock(side_effect=lambda p: p.read_text(encoding="utf-8"))

        def query(_):
            barrier.wait(timeout=5)
            return data_service._cached(path, loader)

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(query, range(8))), ["cached"] * 8)
        loader.assert_called_once()

    def test_file_changed_during_read_is_not_cached(self):
        path = self.root / "sample.txt"
        path.write_text("old", encoding="utf-8")

        def loader(p):
            text = p.read_text(encoding="utf-8")
            p.write_text("new content", encoding="utf-8")
            return text

        with self.assertRaises(data_service.ArtifactUnavailableError):
            data_service._cached(path, loader)
        self.assertNotIn(str(path), data_service._cache)

    def test_prediction_boundaries(self):
        self.write_predictions()
        for date in ("NaT", "invalid", "2026-02-30", "2026-09-10T00:00:00+08:00", "2020-01-01"):
            with self.subTest(date=date), self.assertRaises(ValueError):
                data_service.get_picks(date, 1)
        for topk in (0, -1, 51):
            with self.assertRaises(ValueError):
                data_service.get_picks(None, topk)

    def test_empty_predictions_have_no_dates(self):
        self.write_predictions().iloc[:0].to_pickle(data_service._PRED_FILE)
        self.assertEqual(data_service.available_dates(), [])
        with self.assertRaisesRegex(ValueError, "没有有效打分"):
            data_service.get_picks(None, 1)

    def test_nonfinite_scores_are_excluded(self):
        self.write_predictions([0.1, 0.2, np.inf, np.nan])
        result = data_service.get_picks(None, 2)
        self.assertEqual(result["date"], "2026-09-10")
        self.assertEqual(len(result["rows"]), 2)
        json.dumps(result, allow_nan=False)

    def test_invalid_run_meta_is_an_empty_mapping(self):
        for content in ("[]", "null", "{", '{"tradable":[]}'):
            data_service._RUN_META_FILE.write_text(content, encoding="utf-8")
            self.assertIsInstance(data_service.read_run_meta(), dict)

    def test_backtest_summary_uses_compound_nav(self):
        df = self.write_report([0.01, -0.02, 0.03], [0.002, -0.005, 0.01])
        result = data_service.backtest_summary()
        net = df["return"] - df["cost"]
        excess = net - df["bench"]
        cumulative = (1 + net).cumprod()
        self.assertEqual(result["annualization_days"], 252)
        years = (df.index[-1] - df.index[0]).days / 365.25
        self.assertAlmostEqual(result["strategy_annual"], cumulative.iloc[-1] ** (1/years) - 1, places=6)
        self.assertAlmostEqual(result["strategy_cum_return"], (1 + net).prod() - 1, places=6)
        self.assertAlmostEqual(result["information_ratio"], excess.mean() / excess.std(ddof=1) * np.sqrt(252), places=4)
        self.assertAlmostEqual(result["strategy_max_drawdown"], (cumulative / cumulative.cummax().clip(lower=1) - 1).min(), places=6)

    def test_empty_single_day_and_nonfinite_reports_are_strict_json(self):
        for returns in ([], [0.01], [np.nan, np.inf], [0.01, np.nan, 0.02]):
            with self.subTest(returns=returns):
                self.write_report(returns)
                summary = data_service.backtest_summary()
                json.dumps(summary, allow_nan=False)
                json.dumps(data_service.backtest_series(), allow_nan=False)
                if len(returns) < 2:
                    self.assertIsNone(summary["information_ratio"])
                if not returns:
                    self.assertIsNone(summary["start"])
                    self.assertIsNone(summary["strategy_cum_return"])
                    self.assertEqual(data_service.backtest_series()["dates"], [])


class ApiTests(ArtifactFixture, unittest.IsolatedAsyncioTestCase):
    async def request(self, path, **kwargs):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path, **kwargs)

    async def test_health(self):
        response = await self.request("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_invalid_query_parameters_return_422(self):
        self.write_predictions()
        cases = [("/api/picks", {"date": d}) for d in
                 ("NaT", "invalid", "2026-09-10T00:00:00+08:00")]
        cases += [("/api/picks", {"topk": k}) for k in (-1, 0, 51)]
        cases += [("/api/pipeline/logs/train", {"tail": n}) for n in (-1, 0, 10001)]
        for path, params in cases:
            with self.subTest(params=params):
                response = await self.request(path, params=params)
                self.assertEqual(response.status_code, 422, response.text)

    async def test_invalid_calendar_date_returns_400(self):
        self.write_predictions()
        response = await self.request("/api/picks", params={"date": "2026-02-30"})
        self.assertEqual(response.status_code, 400, response.text)

    async def test_missing_predictions_and_report_return_404(self):
        for path in ("/api/picks", "/api/backtest/report", "/api/backtest/summary"):
            self.assertEqual((await self.request(path)).status_code, 404)

    async def test_empty_tradable_pool_remains_empty(self):
        data_service._RUN_META_FILE.write_text('{"tradable":[]}', encoding="utf-8")
        response = await self.request("/api/universe")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(row["tradable"] for row in response.json()))

    async def test_missing_metadata_does_not_authorize_static_universe(self):
        response = await self.request("/api/universe")
        self.assertFalse(any(row["tradable"] for row in response.json()))

    async def test_corrupt_artifact_returns_retryable_503(self):
        data_service._PRED_FILE.write_bytes(b"incomplete pickle")
        response = await self.request("/api/picks")
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.headers["retry-after"], "2")

    async def test_invalid_report_columns_return_503(self):
        data_service._REPORT_FILE.write_text("date,unexpected\n2026-09-10,0.1\n", encoding="utf-8")
        response = await self.request("/api/backtest/report")
        self.assertEqual(response.status_code, 503, response.text)

    async def test_single_day_summary_returns_null_ir(self):
        self.write_report([0.01])
        response = await self.request("/api/backtest/summary")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["information_ratio"])

    async def test_nontrading_date_falls_back(self):
        self.write_predictions()
        response = await self.request("/api/picks", params={"date": "2026-09-12", "topk": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["date"], "2026-09-11")


if __name__ == "__main__":
    unittest.main()
