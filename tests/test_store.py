"""使用内存 DuckDB 验证失败写入不会删除已有行情。"""
import unittest

import duckdb
import pandas as pd

from src.datasource.store import DataStore


class DataStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = DataStore(":memory:")
        self.addCleanup(self.store.close)
        self.bars = pd.DataFrame([
            {"symbol": "SH510300", "date": "2026-09-10", "close": 3.5},
            {"symbol": "SH510300", "date": "2026-09-11", "close": 3.6},
        ])
        self.store.upsert_bars(self.bars)

    def assert_no_temporary_table(self):
        self.assertEqual(self.store.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = '_tmp'"
        ).fetchall(), [])

    def test_failed_conversion_preserves_rows_and_connection_recovers(self):
        bad = self.bars.assign(close=["invalid", "3.7"])
        with self.assertRaises(duckdb.Error):
            self.store.upsert_bars(bad)
        self.assertEqual(self.store.load_bars_raw()["close"].tolist(), [3.5, 3.6])
        self.assert_no_temporary_table()
        self.store.upsert_bars(self.bars.assign(close=[3.7, 3.8]))
        self.assertEqual(self.store.load_bars_raw()["close"].tolist(), [3.7, 3.8])

    def test_duplicate_keys_roll_back_the_entire_batch(self):
        duplicate = pd.concat([self.bars, self.bars.iloc[:1]], ignore_index=True)
        with self.assertRaises(duckdb.Error):
            self.store.upsert_bars(duplicate)
        self.assertEqual(self.store.load_bars_raw()["close"].tolist(), [3.5, 3.6])
        self.assert_no_temporary_table()

    def test_failed_append_releases_temporary_table(self):
        with self.assertRaises(duckdb.Error):
            self.store.log_ingest([{"rows": "invalid"}])
        self.assert_no_temporary_table()
        self.store.log_ingest([{"rows": 2}])
        self.assertEqual(self.store.con.execute("SELECT rows FROM ingest_log").fetchall(), [(2,)])

    def test_index_upsert_still_updates_existing_rows(self):
        data = pd.DataFrame([{"ts_code": "000300.SH", "date": "2026-09-11", "close": 4000.0}])
        self.store.upsert_index(data)
        self.store.upsert_index(data.assign(close=4100.0))
        self.assertEqual(self.store.load_index("000300.SH")["close"].tolist(), [4100.0])


if __name__ == "__main__":
    unittest.main()
