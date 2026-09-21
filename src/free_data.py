"""无令牌的免费增量更新：逐只验证、暂存库完整验收、备份后原子切换。"""
from __future__ import annotations
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from filelock import FileLock
import config
import universe
from backend.artifacts import write_json_atomic
from src.datasource import DataStore
from src.datasource.free_source import fetch_recent_bars, fetch_actions, aligned_increment
from src.runs import sha256


def run_free_update():
    run_id = "free-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    directory = config.OUTPUT_DIR / "data_updates" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    if not config.DUCKDB_PATH.exists():
        raise ValueError("免费增量需要经过核对的历史库；不能从缺失资料创建已审计全历史")
    with FileLock(str(config.DUCKDB_PATH) + ".ingest.lock"):
        original_hash = sha256(config.DUCKDB_PATH)
        with DataStore(read_only=True) as store:
            old = store.con.execute("SELECT b.*, a.adj_factor FROM bars_raw b JOIN adj_factor a USING(symbol,date)").df()
        symbols = sorted(old["symbol"].unique())
        required = [s for s in symbols if universe.get_asset_class(s[2:]) != "qdii"]
        end = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        def one(symbol):
            try:
                raw, actions = fetch_recent_bars(symbol, end), fetch_actions(symbol)
                raw.to_parquet(directory / f"{symbol}_raw.parquet", index=False)
                actions.to_parquet(directory / f"{symbol}_actions.parquet", index=False)
                newer, audit = aligned_increment(raw, actions, old[old["symbol"] == symbol])
                return symbol, newer, audit
            except Exception as exc:
                return symbol, None, {"error": str(exc), "error_type": type(exc).__name__}
        outcomes = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol, frame, audit in pool.map(one, symbols):
                outcomes.append((symbol, frame, audit))
                print(f"免费源 {symbol}: {audit}", flush=True)
        report = {"run_id": run_id, "source": "tencent_raw+sina_corporate_actions",
                  "symbols": {s: a for s, _, a in outcomes}, "promoted": False,
                  "nav_note": "保留历史已公告NAV；未用当期规模回填历史。完整公司行动和标的目录审计仍未通过。"}
        write_json_atomic(directory / "update_report.json", report)
        failed = [s for s, frame, audit in outcomes if frame is None and s in required]
        if failed:
            raise ValueError(f"{len(failed)}个境内ETF验证失败，原库未变；明细：{directory}")
        latest = [a["latest"] for s, _, a in outcomes if s in required]
        if len(set(latest)) != 1 or (end - pd.Timestamp(min(latest))).days > config.MAX_DATA_STALE_DAYS:
            raise ValueError("免费数据的标的覆盖或时效不齐，原库未变")
        stage = directory / "market.staged.duckdb"
        shutil.copy2(config.DUCKDB_PATH, stage)
        now = pd.Timestamp.now().isoformat()
        with DataStore(stage) as store:
            for symbol, frame, audit in outcomes:
                if frame is None or frame.empty:
                    continue
                for field, value in (("source", "tencent_raw"), ("fetch_time", now), ("data_version", run_id)):
                    frame[field] = value
                bars = frame.drop(columns="adj_factor")
                adj = frame[["symbol", "date", "adj_factor", "source", "fetch_time", "data_version"]].copy()
                adj["source"] = "sina_actions_anchored"
                store.upsert_bars(bars)
                store.upsert_adj(adj)
                store.log_ingest([dict(run_id=run_id, source="tencent_raw+sina_actions", interface="verified_free_increment",
                    symbol=symbol, rows=len(frame), date_min=str(frame["date"].min().date()),
                    date_max=str(frame["date"].max().date()), fetch_time=now, data_version=run_id,
                    status="ok", error="")])
            store.con.execute("CHECKPOINT")
        from src.market import load_market
        market = load_market(stage)
        if sha256(config.DUCKDB_PATH) != original_hash or Path(str(config.DUCKDB_PATH) + ".wal").exists():
            raise ValueError("原库在更新期间被其他任务修改，停止切换")
        backup = directory / "market.previous.duckdb"
        shutil.copy2(config.DUCKDB_PATH, backup)
        os.replace(stage, config.DUCKDB_PATH)
        report.update(promoted=True, market_end=market.audit["end"], backup=str(backup),
                      rows_added=sum(len(f) for _, f, _ in outcomes if f is not None))
        write_json_atomic(directory / "update_report.json", report)
        write_json_atomic(config.OUTPUT_DIR / "latest_data_update.json", report)
        print(f"免费增量已验证并切换：{report['rows_added']}行，行情至{report['market_end']}，原库备份{backup}", flush=True)
        return report


if __name__ == "__main__":
    run_free_update()
