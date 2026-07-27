"""Tushare 主源：ETF 日线(fund_daily)、复权因子(fund_adj)、净值/规模(fund_nav)、
元数据(etf_basic，8000 积分不足时降级 fund_basic)。

需环境变量 TUSHARE_TOKEN（见 config.TUSHARE_TOKEN）。返回统一规范化 schema（见 base.py）。
"""
from __future__ import annotations

import time

import pandas as pd
import tushare as ts

import config
import universe
from src.utils import to_qlib_symbol, to_ts_code

_pro = None


def _api():
    global _pro
    if _pro is None:
        if not config.TUSHARE_TOKEN:
            raise RuntimeError("未设置 TUSHARE_TOKEN 环境变量；请先 $env:TUSHARE_TOKEN=\"xxxx\"")
        ts.set_token(config.TUSHARE_TOKEN)
        _pro = ts.pro_api()
    return _pro


def _retry(fn, tries: int = 3, pause: float = 0.4):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:  # 限频/网络抖动，退避重试
            msg = str(e)
            # 积分/权限不足重试无意义，直接抛出（上层会标记该接口跳过）
            if any(k in msg for k in ("权限", "积分", "permission", "没有访问")):
                raise
            last = e
            time.sleep(pause)
    raise last


def _year_chunks(start: str, end: str, span: int = 4):
    """把 [start,end](YYYYMMDD) 切成 <=span 年的窗口，控制单次行数在接口上限内。"""
    s_year, e_year = int(start[:4]), int(end[:4])
    y = s_year
    while y <= e_year:
        lo = f"{max(y, s_year)}0101" if y != s_year else start
        hi_year = min(y + span - 1, e_year)
        hi = f"{hi_year}1231" if hi_year != e_year else end
        yield lo, hi
        y = hi_year + 1


def fetch_bars(code: str, start: str, end: str) -> pd.DataFrame | None:
    """fund_daily 不复权日线 -> 规范化 bars（amount 千元→元、vol 手）。"""
    ts_code = to_ts_code(code)
    symbol = to_qlib_symbol(code)
    if ts_code is None or symbol is None:
        return None
    frames = []
    for s, e in _year_chunks(start, end):
        df = _retry(lambda: _api().fund_daily(ts_code=ts_code, start_date=s, end_date=e))
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(0.12)
    if not frames:
        return None
    raw = pd.concat(frames).drop_duplicates("trade_date")
    out = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw["trade_date"]).dt.strftime("%Y-%m-%d"),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["vol"].astype(float),            # 手
        "amount": raw["amount"].astype(float) * 1000.0,  # 千元 -> 元
        "pct_chg": raw["pct_chg"].astype(float),        # %
    })
    return out.sort_values("date").reset_index(drop=True)


def fetch_adj(code: str, start: str, end: str) -> pd.DataFrame | None:
    """fund_adj 复权因子 -> 规范化 adj。"""
    ts_code = to_ts_code(code)
    symbol = to_qlib_symbol(code)
    if ts_code is None or symbol is None:
        return None
    frames = []
    for s, e in _year_chunks(start, end):
        df = _retry(lambda: _api().fund_adj(ts_code=ts_code, start_date=s, end_date=e))
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(0.12)
    if not frames:
        return None
    raw = pd.concat(frames).drop_duplicates("trade_date")
    out = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(raw["trade_date"]).dt.strftime("%Y-%m-%d"),
        "adj_factor": raw["adj_factor"].astype(float),
    })
    return out.sort_values("date").reset_index(drop=True)


def fetch_nav(code: str, start: str, end: str) -> pd.DataFrame | None:
    """fund_nav 净值/规模 -> 规范化 nav（场内 ETF）。"""
    ts_code = to_ts_code(code)
    symbol = to_qlib_symbol(code)
    if ts_code is None or symbol is None:
        return None
    frames = []
    for s, e in _year_chunks(start, end):
        df = _retry(lambda: _api().fund_nav(ts_code=ts_code, start_date=s, end_date=e))
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(0.12)
    if not frames:
        return None
    raw = pd.concat(frames).drop_duplicates("nav_date")
    for col in ("accum_nav", "adj_nav", "net_asset", "ann_date"):
        if col not in raw.columns:
            raw[col] = None
    out = pd.DataFrame({
        "symbol": symbol,
        "nav_date": pd.to_datetime(raw["nav_date"]).dt.strftime("%Y-%m-%d"),
        "ann_date": pd.to_datetime(raw["ann_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "unit_nav": pd.to_numeric(raw["unit_nav"], errors="coerce"),
        "accum_nav": pd.to_numeric(raw["accum_nav"], errors="coerce"),
        "adj_nav": pd.to_numeric(raw["adj_nav"], errors="coerce"),
        "net_asset": pd.to_numeric(raw["net_asset"], errors="coerce"),
    })
    return out.dropna(subset=["nav_date"]).sort_values("nav_date").reset_index(drop=True)


def _meta_raw() -> pd.DataFrame | None:
    """优先 etf_basic(8000积分)，失败降级 fund_basic(market='E')。返回原始表或 None。"""
    try:
        df = _retry(lambda: _api().etf_basic(list_status="L"), tries=2)
        if df is not None and not df.empty:
            df["_iface"] = "etf_basic"
            return df
    except Exception:
        pass
    try:
        df = _retry(lambda: _api().fund_basic(market="E", status="L"), tries=2)
        if df is not None and not df.empty:
            df["_iface"] = "fund_basic"
            return df
    except Exception:
        pass
    return None


def fetch_meta(codes: list[str]) -> pd.DataFrame:
    """一批 ETF 元数据 -> 规范化 meta；接口不可用时用 universe 兜底。"""
    raw = _meta_raw()
    by_ts: dict[str, dict] = {}
    if raw is not None:
        iface = raw["_iface"].iloc[0]
        for _, r in raw.iterrows():
            ts_code = str(r.get("ts_code", "") or "")
            if not ts_code:
                continue
            if iface == "etf_basic":
                by_ts[ts_code] = {
                    "name": r.get("extname") or r.get("csname") or r.get("cname"),
                    "list_date": r.get("list_date"),
                    "etf_type": r.get("etf_type"),
                    "index_code": r.get("index_code"),
                    "mgr_name": r.get("mgr_name"),
                }
            else:  # fund_basic
                by_ts[ts_code] = {
                    "name": r.get("name"),
                    "list_date": r.get("list_date"),
                    "etf_type": r.get("invest_type") or r.get("fund_type"),
                    "index_code": r.get("benchmark"),
                    "mgr_name": r.get("management"),
                }

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for code in codes:
        code = str(code).zfill(6)
        ts_code = to_ts_code(code)
        symbol = to_qlib_symbol(code)
        if ts_code is None or symbol is None:
            continue
        m = by_ts.get(ts_code, {})
        etf_type = m.get("etf_type")
        # QDII 归类：接口标记为 QDII 或本地覆盖为 qdii
        local_ac = universe.get_asset_class(code)
        is_qdii = (isinstance(etf_type, str) and "QDII" in etf_type.upper()) or local_ac == "qdii"
        asset_class = "qdii" if is_qdii else local_ac
        ld = m.get("list_date")
        list_date = pd.to_datetime(str(ld), errors="coerce") if ld else pd.NaT
        rows.append({
            "symbol": symbol,
            "ts_code": ts_code,
            "code": code,
            "name": m.get("name") or universe.UNIVERSE.get(code, ""),
            "exchange": symbol[:2],
            "list_date": list_date.strftime("%Y-%m-%d") if pd.notna(list_date) else None,
            "etf_type": "QDII" if is_qdii else (etf_type or ""),
            "asset_class": asset_class,
            "index_code": m.get("index_code") or "",
            "mgr_name": m.get("mgr_name") or "",
            "updated_at": now,
        })
    return pd.DataFrame(rows)
