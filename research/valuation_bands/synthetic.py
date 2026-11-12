"""确定性合成数据: 仅用于流水线与合成测试, 不得用于任何绩效结论。

生成的数据集写入 ``output/valuation_bands/synthetic/data/<version>/``，
manifest 明确标记 ``dataset_kind=synthetic_fixture``，不进入 ``data/valuation_bands``。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import build_calendar, sha256_file

SYNTHETIC_VERSION = "synthetic-fixture-v1"


def _next_session_19(calendar: pd.DatetimeIndex, dates: pd.Series) -> pd.Series:
    index = {session: pos for pos, session in enumerate(calendar)}
    out = []
    stamp_cache: Dict[pd.Timestamp, pd.Timestamp] = {}
    for date in dates:
        date = pd.Timestamp(date).normalize()
        if date not in stamp_cache:
            pos = index.get(date)
            if pos is None:
                # 非交易日: 取其后第一个交易日
                later = calendar[calendar >= date]
                pos = index[later[0]] if len(later) else len(calendar) - 1
            nxt = calendar[min(pos + 1, len(calendar) - 1)]
            stamp_cache[date] = pd.Timestamp.combine(nxt.date(), datetime.min.time()).replace(hour=19)
        out.append(stamp_cache[date])
    return pd.Series(out, index=dates.index)


def build_synthetic_dataset(root: Path, seed: int = 20261109) -> Path:
    """构造完整合成快照并返回 manifest 路径。"""
    rng = np.random.default_rng(seed)
    calendar = build_calendar("2010-12-01", "2026-11-06")
    sessions = pd.DatetimeIndex(calendar)
    n = len(sessions)

    def ar1(mean: float, phi: float, sigma: float) -> np.ndarray:
        out = np.empty(n)
        out[0] = mean
        shocks = rng.normal(0.0, sigma, n)
        for i in range(1, n):
            out[i] = mean + phi * (out[i - 1] - mean) + shocks[i]
        return out

    # 指数价格: 带期限结构的随机游走(仅用于撮合与基准)
    drift = 0.00025
    shocks = rng.normal(0.0, 0.012, n)
    hs300 = 3000 * np.exp(np.cumsum(drift + shocks - 0.5 * 0.012 ** 2))
    cyb = 1000 * np.exp(np.cumsum(0.00035 + rng.normal(0.0, 0.018, n) - 0.5 * 0.018 ** 2))
    h30269 = 6000 * np.exp(np.cumsum(0.00015 + rng.normal(0.0, 0.009, n) - 0.5 * 0.009 ** 2))

    # 估值必须与价格同源: PE=价格/盈利, PB=价格/净资产, 否则年度价格低点日的估值
    # 会退化为随机抽样, 无法反映"价格低点对应低估值"的策略前提。
    earnings = 100 * np.exp(np.cumsum(0.00027 + rng.normal(0.0, 0.0004, n)))
    book = 100 * np.exp(np.cumsum(0.00031 + rng.normal(0.0, 0.0005, n)))
    pe = np.clip(12.0 * (hs300 / hs300[0]) / (earnings / earnings[0]), 4.0, 40.0)
    pb = np.clip(4.2 * (cyb / cyb[0]) / (book / book[0]), 1.0, 12.0)
    h30269_mean = pd.Series(h30269).rolling(250, min_periods=20).mean().to_numpy()
    dy = np.clip(0.048 * (h30269_mean / h30269), 0.02, 0.09)

    def price_frame(index_id: str, close: np.ndarray, source: str) -> pd.DataFrame:
        frame = pd.DataFrame({
            "index_id": index_id,
            "trade_date": sessions,
            "close": np.round(close, 4),
            "price_type": "price_index_close",
            "source_id": source,
            "raw_sha256": "synthetic",
        })
        frame["effective_available_at"] = _next_session_19(sessions, frame["trade_date"])
        frame["availability_basis"] = "assumed_next_session_19"
        return frame

    index_price = pd.concat([
        price_frame("000300", hs300, "synthetic"),
        price_frame("399006", cyb, "synthetic"),
    ], ignore_index=True)

    def valuation_frame(index_id: str, metric: str, values: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame({
            "index_id": index_id,
            "trade_date": sessions,
            "metric": metric,
            "value": np.round(values, 6),
            "aggregation_method": "总市值加权整体法(合成)",
            "financial_basis": "TTM" if metric == "PE_TTM" else "MRQ",
            "source_id": "synthetic",
            "source_published_at": sessions + pd.Timedelta(hours=17),
            "fetched_at": "2026-11-09T12:00:00+08:00",
            "revision_id": "r0",
            "raw_sha256": "synthetic",
        })
        frame["effective_available_at"] = _next_session_19(sessions, frame["trade_date"])
        frame["availability_basis"] = "assumed_next_session_19"
        return frame

    index_valuation = pd.concat([
        valuation_frame("000300", "PE_TTM", pe),
        valuation_frame("399006", "PB_MRQ", pb),
    ], ignore_index=True)

    def etf_frame(symbol: str, base_close: np.ndarray, listed_from: Optional[str], sigma: float) -> pd.DataFrame:
        mask = np.ones(n, dtype=bool)
        if listed_from is not None:
            mask = sessions >= pd.Timestamp(listed_from)
        idx = np.where(mask)[0]
        close = base_close[idx] * (1 + rng.normal(0, 0.0015, len(idx)))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1 + rng.normal(0, 0.001, len(idx)))
        spread = np.abs(rng.normal(0, 0.002, len(idx)))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        amount = np.round(rng.uniform(8e7, 4e8, len(idx)), 2)
        volume = np.round(amount / close / 100.0, 0) * 100  # 份
        frame = pd.DataFrame({
            "symbol": symbol,
            "trade_date": sessions[idx],
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume_shares": volume,
            "amount_cny": amount,
            "source_id": "synthetic",
            "raw_sha256": "synthetic",
        })
        frame["available_at"] = frame["trade_date"] + pd.Timedelta(hours=16)
        return frame

    etf_bars = pd.concat([
        etf_frame("SH510300", hs300 / 1000.0, None, 0.002),
        etf_frame("SZ159915", cyb / 1000.0, "2011-12-09", 0.002),
        etf_frame("SH512890", h30269 / 6000.0, "2019-01-02", 0.002),
    ], ignore_index=True)

    def dividends(symbol: str, every: str, rate: float) -> pd.DataFrame:
        frame = etf_bars[etf_bars["symbol"] == symbol].copy()
        if every == "annual":
            window = frame[frame["trade_date"].dt.strftime("%m-%d").between("12-10", "12-20")]
            chosen = window.groupby(window["trade_date"].dt.year).head(1)
        else:
            chosen = frame[frame["trade_date"].dt.month.isin([3, 6, 9, 12])].groupby(
                [frame["trade_date"].dt.year, frame["trade_date"].dt.month]).head(1)
        rows = []
        for i, row in enumerate(chosen.itertuples(index=False)):
            ex = pd.Timestamp(row.trade_date)
            pos = sessions.get_loc(ex)
            record = sessions[max(0, pos - 1)]
            pay = sessions[min(len(sessions) - 1, pos + 5)]
            rows.append({
                "event_id": f"{symbol}-D{i:03d}",
                "symbol": symbol,
                "type": "dividend",
                "announced_at": ex - pd.Timedelta(days=12),
                "record_date": record,
                "ex_date": ex,
                "payment_date": pay,
                "cash_per_share": round(float(row.close) * rate, 6),
                "share_factor": 1.0,
                "evidence_sha256": "synthetic",
            })
        return pd.DataFrame(rows)

    etf_actions = pd.concat([
        dividends("SH510300", "annual", 0.012),
        dividends("SZ159915", "annual", 0.006),
        dividends("SH512890", "quarterly", 0.011),
    ], ignore_index=True)

    ratio_dates = sessions[sessions >= pd.Timestamp("2014-12-01")]
    tr_ret = rng.normal(0.00025, 0.012, len(ratio_dates))
    bond_ret = rng.normal(0.00012, 0.0012, len(ratio_dates))
    tr = 1000 * np.exp(np.cumsum(tr_ret - 0.5 * 0.012 ** 2))
    bond = 1000 * np.exp(np.cumsum(bond_ret - 0.5 * 0.0012 ** 2))
    ratio_signals = pd.DataFrame({
        "trade_date": ratio_dates,
        "tr300": np.round(tr, 6),
        "bond_index": np.round(bond, 6),
        "source_id": "synthetic",
    })
    base_mask = ratio_signals["trade_date"] == pd.Timestamp("2015-01-05")
    if base_mask.sum() == 0:
        # 保证固定基日存在真实观测
        extra = pd.DataFrame([{
            "trade_date": pd.Timestamp("2015-01-05"), "tr300": 1000.0,
            "bond_index": 1000.0, "source_id": "synthetic",
        }])
        ratio_signals = pd.concat([ratio_signals, extra], ignore_index=True).sort_values("trade_date")
    ratio_signals["effective_available_at"] = _next_session_19(sessions, ratio_signals["trade_date"])
    ratio_signals["availability_basis"] = "assumed_next_session_19"

    dy_start = sessions[sessions >= pd.Timestamp("2018-12-03")]
    dividend_yield = pd.DataFrame({
        "index_id": "H30269",
        "trade_date": dy_start,
        "yield_ttm": np.round(dy[-len(dy_start):], 6),
        "source_id": "synthetic",
        "aggregation_method": "指数整体加权TTM股息率(合成)",
    })
    dividend_yield["effective_available_at"] = _next_session_19(sessions, dividend_yield["trade_date"])
    dividend_yield["availability_basis"] = "assumed_next_session_19"

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "index_price": index_price,
        "index_valuation": index_valuation,
        "etf_bars": etf_bars,
        "etf_actions": etf_actions,
        "ratio_signals": ratio_signals,
        "dividend_yield": dividend_yield,
    }
    entries: List[Dict[str, object]] = []
    for name, frame in files.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        entries.append({
            "name": name,
            "path": path.name,
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        })

    manifest = {
        "data_version": SYNTHETIC_VERSION,
        "dataset_kind": "synthetic_fixture",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "calendar": {"exchange": "XSHG", "start": str(sessions[0].date()), "end": str(sessions[-1].date())},
        "files": entries,
        "sources": [{"series": "*", "provider": "synthetic", "availability_basis": "assumed_next_session_19",
                     "evidence_grade": "synthetic_fixture", "pit_verified": False}],
        "assumptions": [
            "全部数值由固定随机种子生成, 不代表任何真实市场数据, 不得用于绩效结论。",
            "仅用于验证数据契约、状态机、账本、对账与 CLI 流程。",
        ],
        "probes": [],
        "usage": "synthetic_pipeline_validation_only",
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
