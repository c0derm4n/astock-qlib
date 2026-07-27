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
from src.universe_filter import build_dynamic_universe

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


def write_instruments(data: dict[str, pd.DataFrame], tradable: set[str]) -> None:
    """all.txt 含全部(+基准实例，供 benchmark 读取)；<market>.txt 仅含动态可交易 ETF。"""
    inst_dir = config.QLIB_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    all_lines, tradable_lines = [], []
    for symbol, df in data.items():
        line = f"{symbol}\t{df['date'].iloc[0]}\t{df['date'].iloc[-1]}"
        all_lines.append(line)
        if not symbol.startswith("BENCH") and symbol in tradable:
            tradable_lines.append(line)
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

        # 动态可交易池（Phase 5）：上市时长/流动性/规模过滤；关闭则全部可交易
        if getattr(config, "USE_DYNAMIC_UNIVERSE", False):
            tradable = set(build_dynamic_universe(store))
            print(f"动态池：{len(tradable)}/{len([s for s in data if not s.startswith('BENCH')])} 只通过过滤")
        else:
            tradable = {s for s in data if not s.startswith("BENCH")}

        # 清空旧的 features，避免残留(calendar 变化会导致下标错位)
        feat_root = config.QLIB_DATA_DIR / "features"
        if feat_root.exists():
            shutil.rmtree(feat_root)

        calendar = build_calendar(data)
        print(f"全局日历：{len(calendar)} 个交易日（{calendar[0]} ~ {calendar[-1]}）")

        # 基准实例（不可交易，供回测对标）：
        #   BENCH    = 全ETF等权指数（衡量行业/风格轮动alpha）
        #   BENCH300 = 沪深300（以沪深300ETF代理，衡量能否跑赢大盘）
        data[config.EQW_BENCH_SYMBOL] = build_benchmark(data, calendar)
        if "SH510300" in data:
            data[config.BENCHMARK_SYMBOL] = data["SH510300"].copy()
        else:
            print("警告：缺少 SH510300，无法生成沪深300基准，回测需退回等权基准")

        write_calendar(calendar)
        write_instruments(data, tradable)
        write_features(data, calendar)

        # 数据血缘/溯源：记录本次 dump 的数据版本与范围（Phase 3）
        run_meta = {
            "dumped_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_version": store.latest_run_id(),
            "calendar_start": calendar[0],
            "calendar_end": calendar[-1],
            "n_instruments_all": len([s for s in data if not s.startswith("BENCH")]),
            "n_tradable": len(tradable),
            "tradable": sorted(tradable),
        }
        (config.OUTPUT_DIR / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成，Qlib 数据已写入：{config.QLIB_DATA_DIR}")
    print(f"数据溯源写入：{config.OUTPUT_DIR / 'run_meta.json'}")


if __name__ == "__main__":
    main()
