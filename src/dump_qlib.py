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

import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

import config
import universe
from src.utils import to_qlib_symbol

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor", "change"]


def load_all() -> dict[str, pd.DataFrame]:
    """读取当前 ETF 池对应的缓存 CSV，返回 {symbol: df}。

    只加载 universe 中的 ETF 符号，自动忽略 raw_cache 里的历史遗留文件
    (如上一版的个股缓存)，避免污染 ETF 池与合成基准。
    """
    valid = {to_qlib_symbol(c) for c in universe.get_codes()}
    valid.discard(None)
    data = {}
    for f in sorted(config.RAW_CACHE_DIR.glob("*.csv")):
        if f.stem.startswith("_"):  # 跳过 _names.csv 等辅助文件
            continue
        if f.stem not in valid:     # 跳过非当前 ETF 池的历史遗留缓存(如旧个股)
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        data[f.stem] = df
    return data


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


def write_instruments(data: dict[str, pd.DataFrame]) -> None:
    inst_dir = config.QLIB_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    all_lines, tradable_lines = [], []
    for symbol, df in data.items():
        line = f"{symbol}\t{df['date'].iloc[0]}\t{df['date'].iloc[-1]}"
        all_lines.append(line)
        if symbol != config.BENCHMARK_SYMBOL:  # 基准指数不计入可交易股票池
            tradable_lines.append(line)
    # all.txt 含基准指数(供 benchmark 读取)；csi300.txt 仅含可交易股票
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
    data = load_all()
    if not data:
        raise SystemExit(f"没有缓存数据，请先运行 fetch_data。目录：{config.RAW_CACHE_DIR}")
    print(f"读取到 {len(data)} 只股票")

    # 清空旧的 features，避免残留(calendar 变化会导致下标错位)
    feat_root = config.QLIB_DATA_DIR / "features"
    if feat_root.exists():
        shutil.rmtree(feat_root)

    calendar = build_calendar(data)
    print(f"全局日历：{len(calendar)} 个交易日（{calendar[0]} ~ {calendar[-1]}）")

    # 合成等权基准指数，作为不可交易的对标实例
    data[config.BENCHMARK_SYMBOL] = build_benchmark(data, calendar)

    write_calendar(calendar)
    write_instruments(data)
    write_features(data, calendar)

    print(f"完成，Qlib 数据已写入：{config.QLIB_DATA_DIR}")


if __name__ == "__main__":
    main()
