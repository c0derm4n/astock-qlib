"""动态 ETF 池过滤（Phase 5）。

从 DuckDB 的 bars_raw / nav / etf_meta 出发，按三条硬约束筛掉次新、低流动、
小规模标的，得到某时点(as_of)可交易的 ETF 池。默认以最新交易日为 as_of，供
dump_qlib 写入 instruments/<market>.txt。
"""
from __future__ import annotations

import config
from src.datasource import DataStore


def build_dynamic_universe(store: DataStore, as_of: str | None = None,
                           verbose: bool = True) -> list[str]:
    """返回通过过滤的可交易 ETF 符号列表（Qlib 符号，如 SH510300）。

    过滤条件：
    - 上市时长：bars_raw 中 <= as_of 的交易日数 >= UNIVERSE_MIN_LIST_DAYS
    - 流动性：近 20 日日均 amount >= UNIVERSE_MIN_AVG_AMOUNT（元）
    - 规模：nav 最新 net_asset >= UNIVERSE_MIN_AUM（元）；net_asset 缺失时不拦截
    """
    bars = store.load_bars_raw()
    if bars.empty:
        return []
    if as_of is None:
        as_of = str(bars["date"].max())
    bars = bars[bars["date"] <= as_of]

    net = store.latest_net_asset(as_of)
    meta = store.load_meta()
    name_by = dict(zip(meta["symbol"], meta["name"])) if not meta.empty else {}

    min_days = int(config.UNIVERSE_MIN_LIST_DAYS)
    min_amt = float(config.UNIVERSE_MIN_AVG_AMOUNT)
    min_aum = float(config.UNIVERSE_MIN_AUM)

    keep: list[str] = []
    dropped: list[tuple[str, str]] = []
    for sym, g in bars.groupby("symbol"):
        sym = str(sym)
        g = g.sort_values("date")
        if len(g) < min_days:
            dropped.append((sym, f"次新({len(g)}<{min_days}交易日)"))
            continue
        avg_amt = float(g["amount"].tail(20).mean())
        if not (avg_amt >= min_amt):
            dropped.append((sym, f"低流动(日均{avg_amt/1e4:.0f}万<{min_amt/1e4:.0f}万)"))
            continue
        na = net.get(sym)
        if na is not None and not (na >= min_aum):
            dropped.append((sym, f"小规模({na/1e8:.2f}亿<{min_aum/1e8:.2f}亿)"))
            continue
        keep.append(sym)

    if verbose and dropped:
        print(f"动态池剔除 {len(dropped)} 只：")
        for sym, why in dropped:
            print(f"  - {sym} {name_by.get(sym, '')}: {why}")
    return sorted(keep)
