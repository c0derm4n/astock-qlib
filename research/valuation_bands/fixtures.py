"""测试与合成场景的轻量数据构造工具(不属于策略逻辑)。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibrationSnapshot
from .data import ResearchData, build_calendar, sha256_json


def next_session_19(calendar: pd.DatetimeIndex, dates: Iterable) -> List[pd.Timestamp]:
    position = {session: pos for pos, session in enumerate(calendar)}
    out = []
    for date in pd.to_datetime(list(dates)):
        date = pd.Timestamp(date).normalize()
        pos = position.get(date)
        if pos is None:
            later = calendar[calendar >= date]
            pos = position[later[0]] if len(later) else len(calendar) - 1
        nxt = calendar[min(pos + 1, len(calendar) - 1)]
        out.append(pd.Timestamp.combine(nxt.date(), pd.Timestamp("1970-01-01").time()).replace(hour=19))
    return out


def make_dataset(
    start: str,
    end: str,
    price_map: Mapping[str, Sequence[float]],
    valuation_map: Mapping[Tuple[str, str], Sequence[float]],
    etf_map: Mapping[str, Mapping[str, Sequence[float]]],
    actions: Optional[pd.DataFrame] = None,
    ratio: Optional[pd.DataFrame] = None,
    dividend_yield: Optional[pd.DataFrame] = None,
    dataset_kind: str = "synthetic_fixture",
    availability_basis: str = "assumed_next_session_19",
) -> ResearchData:
    """按交易日历对齐构造 ResearchData, 所有观测的可得时间默认下一交易日19:00。"""
    calendar = build_calendar(start, end)
    sessions = pd.DatetimeIndex(calendar)
    n = len(sessions)
    stamps = next_session_19(sessions, sessions)

    rows = []
    for index_id, values in price_map.items():
        array = np.asarray(values, dtype=float)
        assert len(array) == n, f"{index_id} 长度 {len(array)} != 交易日 {n}"
        rows.append(pd.DataFrame({
            "index_id": index_id, "trade_date": sessions, "close": array,
            "price_type": "price_index_close", "source_id": "fixture",
            "effective_available_at": stamps, "availability_basis": availability_basis,
            "raw_sha256": "fixture",
        }))
    index_price = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["index_id", "trade_date", "close", "price_type", "source_id",
                 "effective_available_at", "availability_basis", "raw_sha256"])

    rows = []
    for (index_id, metric), values in valuation_map.items():
        array = np.asarray(values, dtype=float)
        assert len(array) == n
        rows.append(pd.DataFrame({
            "index_id": index_id, "trade_date": sessions, "metric": metric, "value": array,
            "aggregation_method": "fixture", "financial_basis": "TTM",
            "source_id": "fixture", "source_published_at": sessions,
            "effective_available_at": stamps, "availability_basis": availability_basis,
            "fetched_at": "2026-11-09", "revision_id": "r0", "raw_sha256": "fixture",
        }))
    index_valuation = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["index_id", "trade_date", "metric", "value", "aggregation_method", "financial_basis",
                 "source_id", "source_published_at", "effective_available_at", "availability_basis",
                 "fetched_at", "revision_id", "raw_sha256"])

    rows = []
    for symbol, spec in etf_map.items():
        close = np.asarray(spec["close"], dtype=float)
        length = len(close)
        assert length == n
        open_ = np.asarray(spec.get("open", close), dtype=float)
        volume = np.asarray(spec.get("volume", np.full(n, 1e7)), dtype=float)
        amount = np.asarray(spec.get("amount", np.full(n, 2e8)), dtype=float)
        rows.append(pd.DataFrame({
            "symbol": symbol, "trade_date": sessions,
            "open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
            "close": close, "volume_shares": volume, "amount_cny": amount,
            "source_id": "fixture",
            "available_at": sessions + pd.Timedelta(hours=16), "raw_sha256": "fixture",
        }))
    etf_bars = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume_shares",
                 "amount_cny", "source_id", "available_at", "raw_sha256"])

    if actions is None:
        actions = pd.DataFrame(columns=["event_id", "symbol", "type", "announced_at", "record_date",
                                        "ex_date", "payment_date", "cash_per_share", "share_factor",
                                        "evidence_sha256"])
    frames = {
        "index_price": index_price,
        "index_valuation": index_valuation,
        "etf_bars": etf_bars,
        "etf_actions": actions,
        "ratio_signals": ratio if ratio is not None else pd.DataFrame(
            columns=["trade_date", "tr300", "bond_index", "source_id", "effective_available_at",
                     "availability_basis"]),
        "dividend_yield": dividend_yield if dividend_yield is not None else pd.DataFrame(
            columns=["index_id", "trade_date", "yield_ttm", "source_id", "effective_available_at",
                     "availability_basis", "aggregation_method"]),
    }
    manifest = {
        "data_version": "fixture",
        "dataset_kind": dataset_kind,
        "created_at": "2026-11-09T00:00:00+08:00",
        "calendar": {"exchange": "XSHG", "start": start, "end": end},
        "files": [],
        "sources": [],
        "probes": [],
    }
    return ResearchData(root=Path("."), manifest=manifest, frames=frames, calendar=sessions, absences=[])


def make_snapshot(l1: float, l2: float, l3: float, s: float, year: int = 2016,
                  q: float = 0.85, strategy: str = "A", index_id: str = "000300",
                  metric: str = "PE_TTM") -> CalibrationSnapshot:
    payload = {"l1": l1, "l2": l2, "l3": l3, "s": s, "year": year, "q": q}
    return CalibrationSnapshot(
        strategy=strategy, index_id=index_id, metric=metric, year=year,
        lookback_years=list(range(year - 5, year)), q=q,
        lows=[{"year": year - 5 + i, "trade_date": f"{year - 5 + i}-06-30", "close": 1.0,
               "value": l2, "price_coverage": 1.0, "valuation_coverage": 1.0, "tie_count": 1}
              for i in range(5)],
        median_low_value=l2, l1=l1, l2=l2, l3=l3, exit_level=s,
        sample_count=1250, price_coverage={}, valuation_coverage={},
        snapshot_hash=sha256_json(payload), created_at="2026-11-09T00:00:00+08:00",
        params_hash="fixture",
    )


def make_actions(symbol: str, events: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": f"{symbol}-{index}",
        "symbol": symbol,
        "type": event.get("type", "dividend"),
        "announced_at": event.get("announced_at", ""),
        "record_date": event.get("record_date", ""),
        "ex_date": event.get("ex_date"),
        "payment_date": event.get("payment_date", ""),
        "cash_per_share": float(event.get("cash_per_share", 0.0)),
        "share_factor": float(event.get("share_factor", 1.0)),
        "evidence_sha256": "fixture",
    } for index, event in enumerate(events)])


def write_snapshot(directory: Path, frames: Mapping[str, pd.DataFrame], calendar: Tuple[str, str],
                   dataset_kind: str = "synthetic_fixture", data_version: str = "fixture") -> Path:
    """把构造好的数据写成 manifest + CSV 快照, 供 CLI/验收测试使用。"""
    from .data import sha256_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, frame in frames.items():
        if frame is None or frame.empty:
            continue
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        entries.append({"name": name, "path": path.name, "sha256": sha256_file(path), "rows": int(len(frame))})
    manifest = {
        "data_version": data_version,
        "dataset_kind": dataset_kind,
        "created_at": "2026-11-09T00:00:00+08:00",
        "calendar": {"exchange": "XSHG", "start": calendar[0], "end": calendar[1]},
        "files": entries,
        "sources": [],
        "probes": [],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory / "manifest.json"
