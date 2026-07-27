"""Provider 协议：统一各数据源的规范化返回，便于主源/校验源互换。

各 fetch_* 返回的 DataFrame 列约定（内部规范化 schema）：
- bars : symbol, date(YYYY-MM-DD), open, high, low, close, volume(手), amount(元), pct_chg(%)
- adj  : symbol, date, adj_factor
- nav  : symbol, nav_date, ann_date, unit_nav, accum_nav, adj_nav, net_asset
- meta : symbol, ts_code, code, name, exchange, list_date, etf_type, asset_class, index_code, mgr_name
symbol 统一为 Qlib 符号（如 SH510300）。
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class BarSource(Protocol):
    def fetch_bars(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        """抓取单只 ETF 的不复权日线（6 位代码，start/end 为 YYYYMMDD）。"""
        ...


class AdjSource(Protocol):
    def fetch_adj(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        """抓取单只 ETF 的复权因子。"""
        ...


class NavSource(Protocol):
    def fetch_nav(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        """抓取单只 ETF 的净值/规模。"""
        ...


class MetaSource(Protocol):
    def fetch_meta(self, codes: list[str]) -> pd.DataFrame:
        """抓取一批 ETF 的元数据（上市日/类型/跟踪指数等）。"""
        ...
