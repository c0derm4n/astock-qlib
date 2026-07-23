"""
第 1 步：用 AkShare 抓取 A股 ETF 日线行情，缓存为规范化 CSV。

- ETF 池 = 内置宽基 + 行业主题 ETF(见 universe.py)
- 价格用前复权(qfq)，处理分红对齐；factor 恒为 1
- 每只 ETF 缓存到 raw_cache/<SYMBOL>.csv，已存在则跳过(可 --force 重抓)

用法:
    python -m src.fetch_data            # 抓取(增量，跳过已缓存)
    python -m src.fetch_data --force    # 强制全部重抓
"""
from __future__ import annotations

import argparse
import socket
import time

import akshare as ak
import pandas as pd

import config
import universe
from src.utils import to_qlib_symbol

# 连接/读取超时，避免网络不稳时长时间挂起
socket.setdefaulttimeout(12)


def get_universe() -> list[str]:
    """返回 ETF 池的6位代码列表（内置 ETF 清单）。"""
    codes = universe.get_codes()
    if config.MAX_STOCKS:
        codes = codes[: config.MAX_STOCKS]
    return codes


_RENAME = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
}


def _normalize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """把 AkShare 行情表规范化为 Qlib 所需字段。"""
    df = df.rename(columns=_RENAME)
    need = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"{tag} 缺少列 {missing}，实际列：{list(df.columns)}")
    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # Qlib 所需字段
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0  # 典型价，与qfq一致
    df["factor"] = 1.0                       # 已前复权，因子恒为1
    df["change"] = df["pct_chg"] / 100.0     # 涨跌幅(分数)，用于涨跌停判定
    out = df[["date", "open", "high", "low", "close", "volume", "vwap", "factor", "change"]]
    return out.sort_values("date").reset_index(drop=True)


def fetch_one(code: str, tries: int = 3) -> pd.DataFrame | None:
    """抓取单只 ETF 的前复权日线，失败自动重试(网络间歇不稳)。"""
    last_err = None
    for attempt in range(tries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                adjust="qfq",
            )
            if df is None or df.empty:
                return None
            return _normalize(df, code)
        except Exception as e:  # 连接超时等，短暂休息后重试
            last_err = e
            time.sleep(0.5)
    raise last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重抓已缓存的股票")
    args = parser.parse_args()

    codes = get_universe()
    print(f"ETF池(宽基+行业主题)：共 {len(codes)} 只")

    ok, skip, fail = 0, 0, 0
    for i, code in enumerate(codes, 1):
        symbol = to_qlib_symbol(code)
        if symbol is None:
            continue
        cache_file = config.RAW_CACHE_DIR / f"{symbol}.csv"
        if cache_file.exists() and not args.force:
            skip += 1
            continue
        try:
            df = fetch_one(code)
            if df is None or df.empty:
                fail += 1
                print(f"[{i}/{len(codes)}] {symbol} 无数据，跳过")
                continue
            df.to_csv(cache_file, index=False, encoding="utf-8")
            ok += 1
            if i % 20 == 0 or i == len(codes):
                print(f"[{i}/{len(codes)}] 已抓取 {symbol}（{len(df)} 行）")
            time.sleep(0.2)  # 轻微限速，友好对待数据源
        except Exception as e:  # 单只失败不影响整体
            fail += 1
            print(f"[{i}/{len(codes)}] {symbol} 失败：{e}")
            time.sleep(0.5)

    print(f"\n完成：新增 {ok}，跳过(已缓存) {skip}，失败 {fail}")
    print(f"缓存目录：{config.RAW_CACHE_DIR}")


if __name__ == "__main__":
    main()
