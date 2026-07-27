"""AKShare 数据源：ETF 日线（东财不复权优先，不可达时降级新浪前复权）+ 复权因子。

东财 fund_etf_hist_em 在部分网络环境会连接被重置(RemoteDisconnected)；本模块自动
降级到新浪 fund_etf_hist_sina（前复权），保证 akshare 主源可用：
- 东财可达：fetch_bars 返回不复权原价；fetch_adj 由后复权/不复权反推因子（真后复权）。
- 东财不可达：fetch_bars 返回新浪前复权序列；fetch_adj 记 adj_factor=1（价格已复权）。
两种情况下 dump 的 hfq = 价格 × adj_factor 都得到一致可用的复权序列。
"""
from __future__ import annotations

import socket
import time

import akshare as ak
import pandas as pd

import universe
from src.utils import to_ak_code, to_qlib_symbol

socket.setdefaulttimeout(15)

_EM_DOWN = False               # 本进程内是否已判定东财不可达（避免逐只反复重试拖慢）
_LAST_SRC: dict[str, str] = {}  # symbol -> "em_raw" | "sina"，供 fetch_adj 判断复权口径

_EM_RENAME = {
    "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg",
}
_NEED = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    return df[(df["date"] >= s) & (df["date"] <= e)]


def _em_bars(ak_code: str, start: str, end: str, adjust: str) -> pd.DataFrame | None:
    """东财 fund_etf_hist_em -> 规范化 bars。"""
    df = ak.fund_etf_hist_em(symbol=ak_code, period="daily",
                             start_date=start, end_date=end, adjust=adjust)
    if df is None or df.empty:
        return None
    df = df.rename(columns=_EM_RENAME)
    if any(c not in df.columns for c in _NEED):
        raise RuntimeError(f"东财返回列不含 {_NEED}，实际 {list(df.columns)}")
    return pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),   # 手
        "amount": df["amount"].astype(float),     # 元
        "pct_chg": df["pct_chg"].astype(float),   # %
    })


def _sina_bars(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """新浪 fund_etf_hist_sina（前复权，返回全历史）-> 规范化 bars 并按日期过滤。"""
    df = ak.fund_etf_hist_sina(symbol=symbol.lower())
    if df is None or df.empty:
        return None
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": df["volume"].astype(float) / 100.0,  # 股 -> 手，与东财/Tushare 对齐
        "amount": df["amount"].astype(float),           # 元
    }).sort_values("date").reset_index(drop=True)
    out["pct_chg"] = out["close"].pct_change() * 100.0
    return _in_range(out, start, end).reset_index(drop=True)


def fetch_bars(code: str, start: str, end: str, tries: int = 3) -> pd.DataFrame | None:
    """不复权原价（东财）优先；东财不可达时降级新浪前复权。"""
    global _EM_DOWN
    symbol = to_qlib_symbol(code)
    if symbol is None:
        return None
    ak_code = to_ak_code(code)
    if not _EM_DOWN:
        for _ in range(tries):
            try:
                out = _em_bars(ak_code, start, end, "")
                if out is None:
                    return None  # 空数据（如次新未上市）不代表东财不可用
                _LAST_SRC[symbol] = "em_raw"
                out.insert(0, "symbol", symbol)
                return out.sort_values("date").reset_index(drop=True)
            except Exception:
                time.sleep(0.4)
        _EM_DOWN = True
        print("  [降级] 东财不可达，改用新浪(前复权)源")
    out = _sina_bars(symbol, start, end)
    if out is None or out.empty:
        return None
    _LAST_SRC[symbol] = "sina"
    out.insert(0, "symbol", symbol)
    return out


def fetch_adj(code: str, start: str, end: str, tries: int = 3) -> pd.DataFrame | None:
    """复权因子。东财源：hfq/raw 反推（真后复权）；新浪源：因子=1（价格已前复权）。"""
    symbol = to_qlib_symbol(code)
    if symbol is None:
        return None
    raw = fetch_bars(code, start, end)
    if raw is None or raw.empty:
        return None

    def _ones() -> pd.DataFrame:
        return pd.DataFrame({"symbol": symbol, "date": raw["date"].to_numpy(), "adj_factor": 1.0})

    if _LAST_SRC.get(symbol) == "sina":
        return _ones()

    ak_code = to_ak_code(code)
    hf = None
    for _ in range(tries):
        try:
            hf = ak.fund_etf_hist_em(symbol=ak_code, period="daily",
                                     start_date=start, end_date=end, adjust="hfq")
            break
        except Exception:
            time.sleep(0.4)
    if hf is None or hf.empty:
        return _ones()  # 拿不到后复权则退化为因子=1（等价不做分红调整）
    hf = hf.rename(columns={"日期": "date", "收盘": "hfq_close"})
    hf["date"] = pd.to_datetime(hf["date"]).dt.strftime("%Y-%m-%d")
    m = raw[["date", "close"]].merge(hf[["date", "hfq_close"]], on="date")
    m = m[m["close"] > 0]
    if m.empty:
        return _ones()
    return pd.DataFrame({
        "symbol": symbol,
        "date": m["date"].to_numpy(),
        "adj_factor": m["hfq_close"].astype(float).to_numpy() / m["close"].astype(float).to_numpy(),
    })


def fetch_meta(codes: list[str]) -> pd.DataFrame:
    """元数据降级：仅取名称（AKShare 无稳定上市日/QDII 字段），其余用 universe 兜底。"""
    names: dict[str, str] = {}
    try:
        df = ak.fund_name_em()
        col_code = "基金代码" if "基金代码" in df.columns else df.columns[0]
        col_name = "基金简称" if "基金简称" in df.columns else df.columns[1]
        names = {str(r[col_code]).zfill(6): str(r[col_name]) for _, r in df.iterrows()}
    except Exception:
        pass
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for code in codes:
        code = str(code).zfill(6)
        symbol = to_qlib_symbol(code)
        if symbol is None:
            continue
        ac = universe.get_asset_class(code)
        rows.append({
            "symbol": symbol, "ts_code": "", "code": code,
            "name": names.get(code) or universe.UNIVERSE.get(code, ""),
            "exchange": symbol[:2], "list_date": None,
            "etf_type": "QDII" if ac == "qdii" else "",
            "asset_class": ac, "index_code": "", "mgr_name": "", "updated_at": now,
        })
    return pd.DataFrame(rows)
