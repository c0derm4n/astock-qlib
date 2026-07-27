"""DuckDB 规范化真源 + 数据血缘。

统一存储不复权日线(bars_raw)、复权因子(adj_factor)、净值/规模(nav)、ETF 元数据
(etf_meta)，以及血缘表(ingest_log 拉取 / validation_log 校验)。对上层提供后复权
读取(load_hfq_bars)与溢价折价读取(load_premium)等派生视图。
"""
from __future__ import annotations

import pandas as pd
import duckdb

import config

# 建表 DDL（date/timestamp 列由 upsert 时按需 CAST）
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS etf_meta(
        symbol VARCHAR PRIMARY KEY, ts_code VARCHAR, code VARCHAR, name VARCHAR,
        exchange VARCHAR, list_date DATE, etf_type VARCHAR, asset_class VARCHAR,
        index_code VARCHAR, mgr_name VARCHAR, updated_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS bars_raw(
        symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
        volume DOUBLE, amount DOUBLE, pct_chg DOUBLE, source VARCHAR,
        fetch_time TIMESTAMP, data_version VARCHAR, PRIMARY KEY(symbol, date))""",
    """CREATE TABLE IF NOT EXISTS adj_factor(
        symbol VARCHAR, date DATE, adj_factor DOUBLE, source VARCHAR,
        fetch_time TIMESTAMP, data_version VARCHAR, PRIMARY KEY(symbol, date))""",
    """CREATE TABLE IF NOT EXISTS nav(
        symbol VARCHAR, nav_date DATE, ann_date DATE, unit_nav DOUBLE, accum_nav DOUBLE,
        adj_nav DOUBLE, net_asset DOUBLE, source VARCHAR, fetch_time TIMESTAMP,
        data_version VARCHAR, PRIMARY KEY(symbol, nav_date))""",
    """CREATE TABLE IF NOT EXISTS ingest_log(
        run_id VARCHAR, source VARCHAR, interface VARCHAR, symbol VARCHAR, rows INTEGER,
        date_min VARCHAR, date_max VARCHAR, fetch_time TIMESTAMP, data_version VARCHAR,
        status VARCHAR, error VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS validation_log(
        run_id VARCHAR, symbol VARCHAR, check_type VARCHAR, n_checked INTEGER,
        n_mismatch INTEGER, max_rel_diff DOUBLE, passed BOOLEAN, run_time TIMESTAMP, note VARCHAR)""",
]


class DataStore:
    """DuckDB 单文件库封装（建表 + upsert + 血缘 + 派生读取）。"""

    def __init__(self, path=None):
        self.con = duckdb.connect(str(path or config.DUCKDB_PATH))
        for ddl in _SCHEMA:
            self.con.execute(ddl)

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------ upsert
    def _upsert(self, table: str, df: pd.DataFrame, keys, cast_map=None) -> None:
        """DELETE 命中主键 + INSERT（等价 upsert）。cast_map: {列: SQL类型}。"""
        if df is None or len(df) == 0:
            return
        cast_map = cast_map or {}
        cols = list(df.columns)

        def expr(c: str) -> str:
            return f"CAST({c} AS {cast_map[c]}) AS {c}" if c in cast_map else c

        self.con.register("_tmp", df)
        keysel = ", ".join(expr(k) for k in keys)
        self.con.execute(
            f"DELETE FROM {table} WHERE ({', '.join(keys)}) IN (SELECT {keysel} FROM _tmp)"
        )
        self.con.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"SELECT {', '.join(expr(c) for c in cols)} FROM _tmp"
        )
        self.con.unregister("_tmp")

    def _append(self, table: str, df: pd.DataFrame, cast_map=None) -> None:
        if df is None or len(df) == 0:
            return
        cast_map = cast_map or {}
        cols = list(df.columns)

        def expr(c: str) -> str:
            return f"CAST({c} AS {cast_map[c]}) AS {c}" if c in cast_map else c

        self.con.register("_tmp", df)
        self.con.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"SELECT {', '.join(expr(c) for c in cols)} FROM _tmp"
        )
        self.con.unregister("_tmp")

    def upsert_bars(self, df: pd.DataFrame) -> None:
        self._upsert("bars_raw", df, keys=["symbol", "date"],
                     cast_map={"date": "DATE", "fetch_time": "TIMESTAMP"})

    def upsert_adj(self, df: pd.DataFrame) -> None:
        self._upsert("adj_factor", df, keys=["symbol", "date"],
                     cast_map={"date": "DATE", "fetch_time": "TIMESTAMP"})

    def upsert_nav(self, df: pd.DataFrame) -> None:
        self._upsert("nav", df, keys=["symbol", "nav_date"],
                     cast_map={"nav_date": "DATE", "ann_date": "DATE", "fetch_time": "TIMESTAMP"})

    def upsert_meta(self, df: pd.DataFrame) -> None:
        self._upsert("etf_meta", df, keys=["symbol"],
                     cast_map={"list_date": "DATE", "updated_at": "TIMESTAMP"})

    def log_ingest(self, rows: list[dict]) -> None:
        self._append("ingest_log", pd.DataFrame(rows), cast_map={"fetch_time": "TIMESTAMP"})

    def log_validation(self, rows: list[dict]) -> None:
        self._append("validation_log", pd.DataFrame(rows), cast_map={"run_time": "TIMESTAMP"})

    # ------------------------------------------------------------------- reads
    def bar_date_min(self, symbol: str) -> str | None:
        r = self.con.execute(
            "SELECT min(date) FROM bars_raw WHERE symbol = ?", [symbol]
        ).fetchone()
        return r[0].strftime("%Y-%m-%d") if r and r[0] is not None else None

    def bar_date_max(self, symbol: str) -> str | None:
        r = self.con.execute(
            "SELECT max(date) FROM bars_raw WHERE symbol = ?", [symbol]
        ).fetchone()
        return r[0].strftime("%Y-%m-%d") if r and r[0] is not None else None

    def symbols_with_bars(self) -> list[str]:
        return [r[0] for r in self.con.execute(
            "SELECT DISTINCT symbol FROM bars_raw ORDER BY symbol").fetchall()]

    def load_bars_raw(self, symbols=None) -> pd.DataFrame:
        """不复权原始日线（date 为 YYYY-MM-DD 字符串），供交叉校验/流动性统计。"""
        where = ""
        params: list = []
        if symbols:
            where = "WHERE symbol IN (" + ",".join(["?"] * len(symbols)) + ")"
            params = list(symbols)
        df = self.con.execute(
            "SELECT symbol, date, open, high, low, close, volume, amount, pct_chg "
            f"FROM bars_raw {where} ORDER BY symbol, date", params).df()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df

    def load_hfq_bars(self, symbols=None) -> dict[str, pd.DataFrame]:
        """后复权日线：hfq_price = raw_price × adj_factor（缺 adj 视为 1.0）。

        返回 {symbol: df[date, open, high, low, close, volume, vwap, factor, change]}，
        date 为 YYYY-MM-DD 字符串，factor 恒为 1.0（已预复权），change=原价 pct/100。
        """
        where = ""
        params: list = []
        if symbols:
            where = "WHERE b.symbol IN (" + ",".join(["?"] * len(symbols)) + ")"
            params = list(symbols)
        sql = f"""
            SELECT b.symbol, b.date, b.open, b.high, b.low, b.close,
                   b.volume, b.pct_chg, COALESCE(a.adj_factor, 1.0) AS adj
            FROM bars_raw b
            LEFT JOIN adj_factor a ON b.symbol = a.symbol AND b.date = a.date
            {where}
            ORDER BY b.symbol, b.date
        """
        df = self.con.execute(sql, params).df()
        out: dict[str, pd.DataFrame] = {}
        if df.empty:
            return out
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            adj = g["adj"].to_numpy(dtype="float64")
            o = pd.DataFrame({"date": g["date"]})
            o["open"] = g["open"].to_numpy() * adj
            o["high"] = g["high"].to_numpy() * adj
            o["low"] = g["low"].to_numpy() * adj
            o["close"] = g["close"].to_numpy() * adj
            o["volume"] = g["volume"].to_numpy()
            o["vwap"] = (o["high"] + o["low"] + o["close"]) / 3.0
            o["factor"] = 1.0
            o["change"] = g["pct_chg"].to_numpy() / 100.0
            out[str(sym)] = o
        return out

    def load_premium(self, symbols=None, start=None, end=None) -> pd.Series:
        """溢价折价率 = 市场收盘价 / 单位净值 - 1（asof 对齐 nav_date <= date）。

        返回 Series，MultiIndex (datetime, instrument)，name='premium'。
        """
        where = []
        params: list = []
        if symbols:
            where.append("symbol IN (" + ",".join(["?"] * len(symbols)) + ")")
            params += list(symbols)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        bars = self.con.execute(
            f"SELECT symbol, date, close FROM bars_raw {wsql} ORDER BY symbol, date", params
        ).df()
        navs = self.con.execute(
            f"SELECT symbol, nav_date, unit_nav FROM nav {wsql} ORDER BY symbol, nav_date", params
        ).df()
        if bars.empty or navs.empty:
            return pd.Series(dtype="float64", name="premium")
        bars["date"] = pd.to_datetime(bars["date"])
        navs["nav_date"] = pd.to_datetime(navs["nav_date"])
        parts = []
        for sym, gb in bars.groupby("symbol"):
            gn = navs[navs["symbol"] == sym]
            if gn.empty:
                continue
            gb = gb.sort_values("date")
            gn = gn.sort_values("nav_date")
            m = pd.merge_asof(gb, gn[["nav_date", "unit_nav"]], left_on="date",
                              right_on="nav_date", direction="backward")
            m = m[(m["unit_nav"] > 0)]
            if start is not None:
                m = m[m["date"] >= pd.Timestamp(start)]
            if end is not None:
                m = m[m["date"] <= pd.Timestamp(end)]
            if m.empty:
                continue
            prem = m["close"].to_numpy() / m["unit_nav"].to_numpy() - 1.0
            idx = pd.MultiIndex.from_arrays([m["date"].to_numpy(), [sym] * len(m)],
                                            names=["datetime", "instrument"])
            parts.append(pd.Series(prem, index=idx, name="premium"))
        if not parts:
            return pd.Series(dtype="float64", name="premium")
        return pd.concat(parts).sort_index()

    def latest_premium(self, symbols=None) -> dict[str, float]:
        """各标的最新可得溢价率（用于选ETF清单展示）。"""
        s = self.load_premium(symbols)
        if s.empty:
            return {}
        latest = {}
        for (dt, inst), v in s.items():
            latest[inst] = float(v)  # 已按 datetime 升序，最后写入即最新
        return latest

    def load_meta(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM etf_meta").df()

    def latest_net_asset(self, as_of=None) -> dict[str, float]:
        """各标的 <= as_of 的最新非空 net_asset（元），用于规模过滤。"""
        where = "WHERE net_asset IS NOT NULL"
        params: list = []
        if as_of:
            where += " AND nav_date <= ?"
            params = [as_of]
        df = self.con.execute(
            f"SELECT symbol, net_asset, nav_date FROM nav {where} ORDER BY symbol, nav_date",
            params).df()
        out: dict[str, float] = {}
        for _, r in df.iterrows():
            out[str(r["symbol"])] = float(r["net_asset"])  # 升序，最后写入即最新
        return out

    def latest_run_id(self) -> str | None:
        r = self.con.execute(
            "SELECT run_id FROM ingest_log ORDER BY fetch_time DESC LIMIT 1").fetchone()
        return r[0] if r else None
