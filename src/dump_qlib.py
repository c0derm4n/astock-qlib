"""
第 2 步：把 raw_cache 里的 CSV 转成 Qlib 二进制数据(calendar/instruments/features)。

Qlib 文件存储格式(已核对 qlib.data.storage.file_storage)：
- calendars/day.txt          : 全局交易日历，每行一个日期，升序
- instruments/<market>.txt   : "代码<TAB>起始日<TAB>结束日"
- features/<代码小写>/<字段>.day.bin :
      小端 float32 数组，首元素=该股在全局日历中的起始下标，其后为各日数值

用法:
    python -m src.dump_qlib
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

import config
from src.datasource import DataStore
from src.universe_filter import universe_from_store, eligibility_intervals
import universe
from src.utils import symbol_to_code

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor", "change"]


def load_all(store: DataStore) -> dict[str, pd.DataFrame]:
    """从 DuckDB 读后复权日线，返回 {symbol: df[date + FIELDS]}。

    后复权在 store.load_hfq_bars() 内完成（hfq=raw×adj_factor，factor 恒 1.0，
    change=原价 pct/100 供涨跌停判定）。仅含 bars_raw 中的 ETF。
    """
    data = store.load_hfq_bars()
    return {s: df for s, df in data.items()
            if df is not None and not df.empty and df["close"].notna().any()}


def build_calendar(data: dict[str, pd.DataFrame]) -> list[str]:
    """全局日历 = 所有股票交易日的并集(升序)。"""
    all_dates: set[str] = set()
    for df in data.values():
        all_dates.update(df["date"].tolist())
    return sorted(all_dates)


def build_benchmark(data: dict[str, pd.DataFrame], calendar: list[str]) -> pd.DataFrame:
    """用股票池合成等权基准指数：每日收益=当日各股简单收益均值。"""
    ret_by_date: dict[str, list[float]] = defaultdict(list)
    for df in data.values():
        d = df["date"].tolist()
        c = df["close"].to_numpy(dtype=float)
        for i in range(1, len(c)):
            if c[i - 1] > 0:
                ret_by_date[d[i]].append(c[i] / c[i - 1] - 1.0)
    level = 1000.0
    rows = []
    for day in calendar:
        rets = ret_by_date.get(day, [])
        r = float(np.mean(rets)) if rets else 0.0
        level *= 1.0 + r
        rows.append((day, level, r))
    bdf = pd.DataFrame(rows, columns=["date", "close", "change"])
    for col in ("open", "high", "low", "vwap"):  # 基准仅回测读 close/change
        bdf[col] = bdf["close"]
    bdf["volume"] = 0.0
    bdf["factor"] = 1.0
    return bdf[["date", "open", "high", "low", "close", "volume", "vwap", "factor", "change"]]


def write_calendar(calendar: list[str]) -> None:
    cal_dir = config.QLIB_DATA_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")


def write_instruments(data: dict[str, pd.DataFrame], tradable: set[str],
                      entry_dates: dict[str, str] | None = None, intervals=None) -> None:
    """all.txt 含全部(+基准实例，供 benchmark 读取)；<market>.txt 仅含动态可交易 ETF。

    entry_dates：{符号 -> 首个可交易日}，上市观察期 point-in-time 生效时把
    <market>.txt 的起始日后移（all.txt 仍保留全量历史供特征回看/研究）。"""
    inst_dir = config.QLIB_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    all_lines, tradable_lines = [], []
    for symbol, df in data.items():
        all_lines.append(f"{symbol}\t{df['date'].iloc[0]}\t{df['date'].iloc[-1]}")
        if not symbol.startswith("BENCH") and symbol in tradable:
            periods = intervals.get(symbol, []) if intervals is not None else [((entry_dates or {}).get(symbol, df["date"].iloc[0]), df["date"].iloc[-1])]
            for start, end in periods:
                tradable_lines.append(f"{symbol}\t{start}\t{end}")
    (inst_dir / "all.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    (inst_dir / f"{config.MARKET}.txt").write_text("\n".join(tradable_lines) + "\n", encoding="utf-8")


def write_features(data: dict[str, pd.DataFrame], calendar: list[str]) -> None:
    cal_index = {d: i for i, d in enumerate(calendar)}
    feat_root = config.QLIB_DATA_DIR / "features"
    feat_root.mkdir(parents=True, exist_ok=True)

    for symbol, df in data.items():
        idx = df["date"].map(cal_index).to_numpy()
        si, ei = int(idx[0]), int(idx[-1])
        length = ei - si + 1
        sym_dir = feat_root / symbol.lower()
        sym_dir.mkdir(parents=True, exist_ok=True)

        pos = idx - si  # 每行在连续数组中的位置
        for field in FIELDS:
            arr = np.full(length, np.nan, dtype="float32")
            arr[pos] = df[field].to_numpy(dtype="float32")
            # 首元素写起始下标，其后写数值(全部小端 float32)
            out = np.hstack([np.float32(si), arr]).astype("<f4")
            out.tofile(sym_dir / f"{field}.day.bin")


def main() -> None:
    with DataStore() as store:
        data = load_all(store)
        if not data:
            raise SystemExit(
                f"DuckDB 无数据，请先运行：python -m src.fetch_data。库：{config.DUCKDB_PATH}")
        print(f"读取到 {len(data)} 只 ETF（后复权）")
        run_dump(store, data)


def run_dump(store: DataStore, data: dict[str, pd.DataFrame]) -> None:
    """完整历史逐日池导出；先生成新目录再切换，保留上一版数据备份。"""
    import uuid
    from backend.artifacts import write_json_atomic

    frame = universe_from_store(store)
    intervals = eligibility_intervals(frame)
    calendar = build_calendar(data)
    if not calendar:
        raise ValueError("没有有效交易日")
    latest = frame[frame["date"] == pd.Timestamp(calendar[-1])]
    current = sorted(latest.loc[latest["eligible"], "symbol"])
    # 全池等权仅为历史兼容展示；经成本/历史资格对齐的基准由research统一生成。
    eligible = frame.pivot(index="date", columns="symbol", values="eligible")
    prices = pd.DataFrame({sym: df.set_index("date")["close"] for sym, df in data.items()})
    prices.index = pd.to_datetime(prices.index)
    returns = prices.pct_change(fill_method=None).where(eligible.shift(1, fill_value=False)).mean(axis=1).fillna(0)
    benchmark = pd.DataFrame({"date": returns.index.strftime("%Y-%m-%d"), "close": (1000*(1+returns).cumprod()).to_numpy()})
    for field in ("open", "high", "low", "vwap"):
        benchmark[field] = benchmark["close"]
    benchmark["volume"], benchmark["factor"], benchmark["change"] = 0.0, 1.0, returns.to_numpy()
    data[config.EQW_BENCH_SYMBOL] = benchmark[["date"] + FIELDS]
    index = store.load_index(getattr(config, "BENCHMARK_INDEX_TS", "000300.SH"))
    if not index.empty:
        index = index[index["date"].isin(calendar)].copy()
        index["vwap"] = index["amount"] / (index["volume"] * 100).replace(0, float("nan"))
        index["factor"], index["change"] = 1., index["pct_chg"] / 100
        data[config.BENCHMARK_SYMBOL] = index[["date"] + FIELDS]
    elif "SH510300" in data:
        data[config.BENCHMARK_SYMBOL] = data["SH510300"].copy()
    destination = config.QLIB_DATA_DIR
    stage = destination.with_name(destination.name + ".build-" + uuid.uuid4().hex[:8])
    if not stage.resolve().is_relative_to(config.BASE_DIR.resolve()):
        raise ValueError("导出目录必须位于项目内")
    try:
        config.QLIB_DATA_DIR = stage
        write_calendar(calendar)
        write_instruments(data, set(intervals), intervals=intervals)
        write_features(data, calendar)
    finally:
        config.QLIB_DATA_DIR = destination
    if destination.exists():
        backup = destination.with_name(destination.name + ".backup-" + uuid.uuid4().hex[:8])
        destination.rename(backup)
        try:
            stage.rename(destination)
        except Exception:
            backup.rename(destination)
            raise
    else:
        stage.rename(destination)
    frame.to_csv(config.OUTPUT_DIR / "historical_universe.csv", index=False)
    write_json_atomic(config.OUTPUT_DIR / "data_meta.json", {
        "dumped_at": pd.Timestamp.now().isoformat(), "data_version": store.latest_run_id(),
        "calendar_start": calendar[0], "calendar_end": calendar[-1],
        "n_instruments_all": frame["symbol"].nunique(), "n_tradable": len(current), "tradable": current,
        "universe_method": "historical_daily_ann_date_strict", "benchmark": "SH510300 total-return proxy",
    })
    print(f"逐日可交易区间：{sum(map(len, intervals.values()))}；最新可交易：{len(current)}")
    print(f"Qlib已完成分阶段导出：{destination}")


if __name__ == "__main__":
    main()
