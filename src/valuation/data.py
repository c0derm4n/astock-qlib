"""研究数据装载: 只读导入、规范化校验、时点可得性与缺口检查。

本模块只读 ``data/valuation_bands/snapshots/<data_version>/`` 下的规范化文件，
不写任何生产数据库。所有信号取数都通过 ``availability`` 列，禁止反向填充。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SESSION_CLOCK = time(19, 0)

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "index_price": [
        "index_id", "trade_date", "close", "price_type", "source_id",
        "effective_available_at", "availability_basis", "raw_sha256",
    ],
    "index_valuation": [
        "index_id", "trade_date", "metric", "value", "aggregation_method",
        "financial_basis", "source_id", "source_published_at", "effective_available_at",
        "availability_basis", "fetched_at", "revision_id", "raw_sha256",
    ],
    "etf_bars": [
        "symbol", "trade_date", "open", "high", "low", "close", "volume_shares",
        "amount_cny", "source_id", "available_at", "raw_sha256",
    ],
    "etf_actions": [
        "event_id", "symbol", "type", "announced_at", "record_date", "ex_date",
        "payment_date", "cash_per_share", "share_factor", "evidence_sha256",
    ],
    "ratio_signals": [
        "trade_date", "tr300", "bond_index", "source_id", "effective_available_at",
        "availability_basis",
    ],
    "dividend_yield": [
        "index_id", "trade_date", "yield_ttm", "source_id", "effective_available_at",
        "availability_basis", "aggregation_method",
    ],
}

OPTIONAL_FILES = ("ratio_signals", "dividend_yield")


# ---------------------------------------------------------------- 通用工具
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_session(value) -> pd.Timestamp:
    return pd.Timestamp(str(value)[:10])


def session_ts(session: pd.Timestamp, clock: time = SESSION_CLOCK) -> pd.Timestamp:
    return pd.Timestamp.combine(pd.Timestamp(session).date(), clock)


def _parse_ts(series: pd.Series) -> pd.Series:
    def local_stamp(value):
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
            return stamp
        except (ValueError, TypeError):
            return pd.NaT
    return pd.to_datetime(series.map(local_stamp), errors="coerce")


# ---------------------------------------------------------------- 交易日历
def build_calendar(start, end, fallback: Optional[Iterable] = None) -> pd.DatetimeIndex:
    """中国交易日历。优先交易所日历, 无库时退回真实观测日期并显式标记。"""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    try:
        import exchange_calendars as xcals

        cal = xcals.get_calendar("XSHG")
        sessions = cal.sessions_in_range(start, end)
        out = pd.DatetimeIndex([pd.Timestamp(s).normalize() for s in sessions])
    except Exception:  # pragma: no cover - 依赖缺失时的降级
        if fallback is None:
            raise
        dates = sorted({pd.Timestamp(d).normalize() for d in fallback if pd.notna(d)})
        out = pd.DatetimeIndex([d for d in dates if start <= d <= end])
    return out[out.to_series().between(start, end)]


# ---------------------------------------------------------------- 数据结构
@dataclass
class ResearchData:
    """只读研究数据容器。所有金额单位为元, 份额单位为份。"""

    root: Path
    manifest: Dict[str, object]
    frames: Dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    absences: List[Dict[str, object]] = field(default_factory=list)

    # -- 基本集合
    @property
    def data_version(self) -> str:
        return str(self.manifest.get("data_version", "unknown"))

    @property
    def dataset_kind(self) -> str:
        return str(self.manifest.get("dataset_kind", "real"))

    def calendar_range(self, start=None, end=None) -> pd.DatetimeIndex:
        cal = self.calendar
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        return cal

    def session_pos(self, session) -> int:
        return int(self.calendar.get_loc(pd.Timestamp(session).normalize()))

    # -- 具体序列
    def index_price(self, index_id: str) -> pd.DataFrame:
        frame = self.frames.get("index_price")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["index_price"])
        return frame[frame["index_id"] == index_id].sort_values("trade_date")

    def index_valuation(self, index_id: str, metric: str) -> pd.DataFrame:
        frame = self.frames.get("index_valuation")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["index_valuation"])
        sel = frame[(frame["index_id"] == index_id) & (frame["metric"] == metric)]
        return sel.sort_values("trade_date")

    def etf_bars(self, symbol: str) -> pd.DataFrame:
        frame = self.frames.get("etf_bars")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["etf_bars"])
        return frame[frame["symbol"] == symbol].sort_values("trade_date")

    def etf_actions(self, symbol: str) -> pd.DataFrame:
        frame = self.frames.get("etf_actions")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["etf_actions"])
        return frame[frame["symbol"] == symbol].sort_values("ex_date")

    def dividend_yield(self, index_id: str) -> pd.DataFrame:
        frame = self.frames.get("dividend_yield")
        if frame is None or frame.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["dividend_yield"])
        return frame[frame["index_id"] == index_id].sort_values("trade_date")

    @property
    def ratio_signals(self) -> pd.DataFrame:
        frame = self.frames.get("ratio_signals")
        if frame is None:
            return pd.DataFrame(columns=REQUIRED_COLUMNS["ratio_signals"])
        return frame.sort_values("trade_date")

    # -- 时点可得性
    def valuation_asof(
        self, index_id: str, metric: str, session, signal_ts: Optional[pd.Timestamp] = None
    ) -> Optional[Dict[str, object]]:
        """返回 signal 时刻可得的最新估值观测, 不做任何填充。"""
        frame = self.index_valuation(index_id, metric)
        if frame.empty:
            return None
        session = pd.Timestamp(session).normalize()
        stamp = signal_ts if signal_ts is not None else session_ts(session)
        usable = frame[(frame["effective_available_at"] <= stamp) & (frame["trade_date"] <= session)]
        if usable.empty:
            return None
        row = usable.iloc[-1]
        lag = int(self.calendar.get_loc(session) - self.calendar.get_loc(row["trade_date"]))
        return {
            "trade_date": row["trade_date"],
            "value": float(row["value"]),
            "available_at": row["effective_available_at"],
            "availability_basis": row["availability_basis"],
            "source_id": row["source_id"],
            "lag_sessions": lag,
        }

    def price_asof(self, symbol: str, session, column: str = "close") -> Optional[Dict[str, object]]:
        frame = self.etf_bars(symbol)
        if frame.empty:
            return None
        session = pd.Timestamp(session).normalize()
        usable = frame[(frame["trade_date"] <= session) & (frame["available_at"] <= session_ts(session))]
        if usable.empty:
            return None
        row = usable.iloc[-1]
        value = row[column]
        if value is None or not np.isfinite(float(value)):
            return None
        return {
            "trade_date": row["trade_date"],
            "value": float(value),
            "volume_shares": float(row["volume_shares"]),
            "lag_sessions": int(self.calendar.get_loc(session) - self.calendar.get_loc(row["trade_date"])),
        }

    def adv20(self, symbol: str, session) -> Optional[float]:
        """信号日已知的过去20个交易日平均成交额(含信号日)。"""
        frame = self.etf_bars(symbol)
        if frame.empty:
            return None
        session = pd.Timestamp(session).normalize()
        expected = self.calendar[self.calendar <= session][-20:]
        usable = frame[(frame["trade_date"].isin(expected)) & (frame["available_at"] <= session_ts(session))]
        if usable["trade_date"].duplicated().any():
            return None
        if len(usable) < 20:
            return None
        amount = usable["amount_cny"].astype(float)
        if not np.isfinite(amount).all() or (amount <= 0).any():
            return None
        return float(amount.mean())

    def is_listed(self, symbol: str, session) -> bool:
        frame = self.etf_bars(symbol)
        if frame.empty:
            return False
        session = pd.Timestamp(session).normalize()
        return bool((frame["trade_date"] <= session).any())

    def eligibility(self, symbol: str, session) -> Dict[str, object]:
        """信号日时点可得的产品资格。"""
        listed = self.is_listed(symbol, session)
        adv = self.adv20(symbol, session) if listed else None
        price = self.price_asof(symbol, session) if listed else None
        tradable = bool(price and price["lag_sessions"] == 0 and price["value"] > 0
                        and np.isfinite(price["volume_shares"]) and price["volume_shares"] > 0)
        eligible = bool(listed and tradable and adv and adv > 0)
        reasons = []
        if not listed:
            reasons.append("not_listed")
        elif not tradable:
            reasons.append("no_trading")
        if listed and adv is None:
            reasons.append("adv20_unavailable")
        return {
            "symbol": symbol,
            "session": str(pd.Timestamp(session).date()),
            "listed": listed,
            "tradable": tradable,
            "data_valid": bool(adv is not None),
            "adv20_cny": adv,
            "eligible": eligible,
            "reason": "+".join(reasons) if reasons else "ok",
        }

    def yield_asof(self, index_id: str, session) -> Optional[Dict[str, object]]:
        frame = self.dividend_yield(index_id)
        if frame.empty:
            return None
        session = pd.Timestamp(session).normalize()
        stamp = session_ts(session)
        usable = frame[(frame["effective_available_at"] <= stamp) & (frame["trade_date"] <= session)]
        if usable.empty:
            return None
        row = usable.iloc[-1]
        return {
            "trade_date": row["trade_date"],
            "yield_ttm": float(row["yield_ttm"]),
            "available_at": row["effective_available_at"],
            "lag_sessions": int(self.calendar.get_loc(session) - self.calendar.get_loc(row["trade_date"])),
        }

    def ratio_asof(self, session) -> Optional[Dict[str, object]]:
        frame = self.ratio_signals
        if frame.empty:
            return None
        session = pd.Timestamp(session).normalize()
        stamp = session_ts(session)
        usable = frame[(frame["effective_available_at"] <= stamp) & (frame["trade_date"] <= session)]
        if usable.empty:
            return None
        row = usable.iloc[-1]
        return {
            "trade_date": row["trade_date"],
            "tr300": float(row["tr300"]),
            "bond_index": float(row["bond_index"]),
            "lag_sessions": int(self.calendar.get_loc(session) - self.calendar.get_loc(row["trade_date"])),
        }


# ---------------------------------------------------------------- 装载
IDENTITY_COLUMNS = {
    "index_id": 6,
    "metric": 0,
    "symbol": 0,
    "event_id": 0,
    "source_id": 0,
    "type": 0,
    "price_type": 0,
    "availability_basis": 0,
    "aggregation_method": 0,
    "financial_basis": 0,
}


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    # 标识列必须按字符串读入, 否则 "000300" 会被推断成整数而丢失前导零
    dtypes = {column: str for column in IDENTITY_COLUMNS if column in REQUIRED_COLUMNS.get(name, [])}
    frame = pd.read_csv(path, dtype=dtypes or None)
    required = REQUIRED_COLUMNS.get(name, [])
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必须字段: {missing}")
    for column, width in IDENTITY_COLUMNS.items():
        if column in frame.columns:
            series = frame[column].astype(str)
            if width:
                series = series.str.replace(r"\.0$", "", regex=True).str.zfill(width)
            frame[column] = series
    for column in ("trade_date", "ex_date", "record_date", "payment_date"):
        if column in frame.columns:
            frame[column] = _parse_ts(frame[column]).dt.normalize()
    for column in ("effective_available_at", "source_published_at", "fetched_at", "announced_at", "available_at"):
        if column in frame.columns:
            frame[column] = _parse_ts(frame[column])
    keys = {"index_price": ["index_id", "trade_date"],
            "index_valuation": ["index_id", "metric", "trade_date"],
            "etf_bars": ["symbol", "trade_date"], "etf_actions": ["event_id"],
            "ratio_signals": ["trade_date"], "dividend_yield": ["index_id", "trade_date"]}
    if name in keys and frame.duplicated(keys[name]).any():
        raise ValueError(f"{path.name} 存在重复主键，须在快照生成前明确选择历史版本")
    return frame


def load_snapshot(manifest_path, verify_hashes: bool = True) -> ResearchData:
    """按 manifest 装载快照, 校验 SHA256 与字段契约。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    frames: Dict[str, pd.DataFrame] = {}
    absences: List[Dict[str, object]] = []

    listed = {entry["name"]: entry for entry in manifest.get("files", [])}
    for name in REQUIRED_COLUMNS:
        entry = listed.get(name)
        if entry is None or not entry.get("path"):
            absences.append({"file": name, "reason": "not_in_manifest"})
            frames[name] = pd.DataFrame(columns=REQUIRED_COLUMNS[name])
            continue
        path = root / entry["path"]
        if not path.exists():
            absences.append({"file": name, "reason": "missing_file", "path": str(path)})
            frames[name] = pd.DataFrame(columns=REQUIRED_COLUMNS[name])
            continue
        if verify_hashes and entry.get("sha256"):
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                raise ValueError(f"{name} SHA256 不匹配: manifest={entry['sha256'][:12]} actual={actual[:12]}")
        frames[name] = _read_csv(path, name)

    # 日历范围由全部可得价格观测决定
    date_pool: List[pd.Timestamp] = []
    for frame in frames.values():
        if "trade_date" in frame.columns and not frame.empty:
            date_pool.append(frame["trade_date"].min())
            date_pool.append(frame["trade_date"].max())
    cal_cfg = manifest.get("calendar", {})
    start = cal_cfg.get("start") or (min(date_pool) if date_pool else pd.Timestamp("2010-01-01"))
    end = cal_cfg.get("end") or (max(date_pool) if date_pool else pd.Timestamp("2026-11-06"))
    fallback = None
    if date_pool:
        parts = [f["trade_date"] for f in frames.values() if "trade_date" in f.columns and not f.empty]
        fallback = pd.concat(parts) if parts else None
    calendar = build_calendar(start, end, fallback=fallback)

    return ResearchData(root=root, manifest=manifest, frames=frames, calendar=calendar, absences=absences)


# ---------------------------------------------------------------- 缺口检查
def _year_coverage(frame: pd.DataFrame, calendar: pd.DatetimeIndex, column="trade_date") -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    if frame.empty:
        return out
    for year in sorted({int(y) for y in calendar.year}):
        sessions = calendar[calendar.year == year]
        if len(sessions) == 0:
            continue
        observed = set(pd.to_datetime(frame[column]).dt.normalize())
        hit = [s for s in sessions if s in observed]
        out[year] = {
            "year": year,
            "sessions": int(len(sessions)),
            "observed": int(len(hit)),
            "coverage": round(len(hit) / len(sessions), 6),
            "first": str(min(hit).date()) if hit else None,
            "last": str(max(hit).date()) if hit else None,
        }
    return out


def _frequency_note(frame: pd.DataFrame) -> Dict[str, object]:
    if frame.empty:
        return {"rows": 0, "median_gap_days": None, "daily_like": False}
    dates = pd.to_datetime(frame["trade_date"]).drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()
    median_gap = float(gaps.median()) if len(gaps) else None
    return {
        "rows": int(len(dates)),
        "first": str(dates.min().date()),
        "last": str(dates.max().date()),
        "median_gap_days": median_gap,
        "daily_like": bool(median_gap is not None and median_gap <= 4.5),
    }


def check_gaps(data: ResearchData, protocol: Mapping[str, object]) -> Dict[str, object]:
    """输出按序列/日期/字段/证据等级组织的缺口报告。"""
    cal = data.calendar
    warmup = protocol.get("warmup_years", [2011, 2012, 2013, 2014, 2015])
    start = pd.Timestamp(f"{min(warmup)}-01-01")
    end = pd.Timestamp(protocol["requested_end"])
    cal_req = cal[(cal >= start) & (cal <= end)]

    gaps: List[Dict[str, object]] = []
    series: List[Dict[str, object]] = []

    def add_gap(series_id, code, detail, blocking=True, evidence=None):
        gaps.append({
            "series": series_id,
            "code": code,
            "detail": detail,
            "blocking": bool(blocking),
            "evidence": evidence or {},
        })

    # 1) 指数价格
    for index_id in protocol["data_requirements"]["index_price"]:
        frame = data.index_price(index_id)
        cov = _year_coverage(frame, cal_req)
        freq = _frequency_note(frame)
        series.append({"series": f"index_price:{index_id}", "frequency": freq, "coverage": cov})
        if frame.empty:
            add_gap(f"index_price:{index_id}", "missing_series", "未取得该指数价格序列")
            continue
        short = {y: c for y, c in cov.items() if c["coverage"] < 1.0}
        if short:
            add_gap(
                f"index_price:{index_id}", "price_coverage_below_100pct",
                f"{len(short)} 个年度存在缺失交易日, 无法保证年度最低日不遗漏",
                evidence={"years": short},
            )

    # 2) 指数估值
    for index_id, metric in protocol["data_requirements"]["index_valuation"]:
        frame = data.index_valuation(index_id, metric)
        cov = _year_coverage(frame, cal_req)
        freq = _frequency_note(frame)
        series.append({"series": f"index_valuation:{index_id}:{metric}", "frequency": freq, "coverage": cov})
        min_cov = float(protocol["calibration"]["valuation_coverage_min"])
        if frame.empty:
            add_gap(
                f"index_valuation:{index_id}:{metric}", "missing_series",
                "未取得该指数整体法日频估值序列, 年度校准与周度信号均无法生成",
                evidence={"probed_sources": data.manifest.get("probes", [])},
            )
            continue
        if not freq["daily_like"]:
            add_gap(
                f"index_valuation:{index_id}:{metric}", "frequency_insufficient",
                f"中位相邻间隔 {freq['median_gap_days']} 天, 非日频, 不满足 H_y 日频观测要求",
                evidence={"frequency": freq},
            )
        short = {y: c for y, c in cov.items() if c["coverage"] < min_cov}
        if short:
            add_gap(
                f"index_valuation:{index_id}:{metric}", "valuation_coverage_below_min",
                f"{len(short)} 个年度估值覆盖率低于 {min_cov:.0%}",
                evidence={"years": short},
            )
        total = int(len(frame))
        if total < int(protocol["calibration"]["min_valuation_observations"]):
            add_gap(
                f"index_valuation:{index_id}:{metric}", "sample_count_insufficient",
                f"总观测 {total} 条 < 最低 {protocol['calibration']['min_valuation_observations']} 条",
            )

    # 3) ETF 行情
    for symbol in protocol["data_requirements"]["etf"]:
        frame = data.etf_bars(symbol)
        freq = _frequency_note(frame)
        series.append({"series": f"etf_bars:{symbol}", "frequency": freq})
        if frame.empty:
            add_gap(f"etf_bars:{symbol}", "missing_series", "未取得该ETF原始日线")
            continue
        need_start = pd.Timestamp(protocol["research_start"])
        if freq["first"] and pd.Timestamp(freq["first"]) > need_start:
            add_gap(
                f"etf_bars:{symbol}", "listing_after_research_start",
                f"ETF 首个可得交易日 {freq['first']} 晚于研究起点 {need_start.date()}",
                blocking=symbol in ("SH510300", "SZ159915"),
                evidence={"first": freq["first"]},
            )

    # 4) 公司行动
    probe_status = {
        str(p.get("series")): str(p.get("status"))
        for p in data.manifest.get("probes", []) if p.get("series")
    }
    for symbol in ("SH510300", "SZ159915", "SH512890"):
        frame = data.etf_actions(symbol)
        if frame.empty:
            confirmed_empty = (data.dataset_kind != "real" or
                               probe_status.get(f"etf_actions:{symbol}") == "ok")
            add_gap(
                f"etf_actions:{symbol}", "no_action_events",
                "数据源确认无分红/拆分事件" if confirmed_empty else "未取得该ETF公司行动证据",
                blocking=not confirmed_empty,
                evidence={"source_probe": probe_status.get(f"etf_actions:{symbol}")},
            )
        else:
            series.append({"series": f"etf_actions:{symbol}", "frequency": {"rows": int(len(frame))}})

    # 5) 可选序列
    ratio = data.ratio_signals
    series.append({"series": "ratio_signals", "frequency": _frequency_note(ratio)})
    if ratio.empty:
        add_gap(
            "ratio_signals", "blocked_optional_data",
            "未取得 H00300 与 885007.WI 同时可得的日频历史, A_pe_ratio_gate 阻塞",
            evidence={"probed_sources": data.manifest.get("probes", [])},
        )
    else:
        base = pd.Timestamp("2015-01-05")
        has_base = bool((ratio["trade_date"] == base).any()) and ratio[ratio["trade_date"] == base].notna().all(axis=None)
        if not has_base:
            add_gap("ratio_signals", "base_date_missing", "固定基日 2015-01-05 缺少两条真实观测")

    dy = data.dividend_yield("H30269")
    series.append({"series": "dividend_yield:H30269", "frequency": _frequency_note(dy)})
    if dy.empty:
        add_gap(
            "dividend_yield:H30269", "blocked_optional_data",
            "未取得 H30269 指数整体 TTM 股息率日频历史, B_pb_dividend 阻塞",
            evidence={"probed_sources": data.manifest.get("probes", [])},
        )
    else:
        cov = _year_coverage(dy, cal_req[cal_req >= pd.Timestamp("2019-01-01")])
        short = {y: c for y, c in cov.items() if c["coverage"] < 0.95}
        if short:
            add_gap("dividend_yield:H30269", "valuation_coverage_below_min", "股息率覆盖率不足", evidence={"years": short})

    blocking = [g for g in gaps if g["blocking"]]
    blocked_ids = {g["series"] for g in blocking}
    primary_ids = {f"index_price:{s['index_id']}" for s in protocol["strategies"].values()}
    primary_ids |= {f"index_valuation:{s['index_id']}:{s['metric']}" for s in protocol["strategies"].values()}
    primary_ids |= {f"{kind}:{s['etf']}" for s in protocol["strategies"].values()
                    for kind in ("etf_bars", "etf_actions")}
    primary_ready = not bool(primary_ids & blocked_ids)
    return {
        "data_version": data.data_version,
        "dataset_kind": data.dataset_kind,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested_window": {"start": str(start.date()), "end": str(end.date())},
        "absences": data.absences,
        "gaps": gaps,
        "series": series,
        "blocking_count": len(blocking),
        "blocking_series": sorted({g["series"] for g in blocking}),
        "can_run_primary": primary_ready,
        "can_run_ratio_gate": primary_ready and "ratio_signals" not in blocked_ids,
        "can_run_dividend": primary_ready and not bool(
            {"dividend_yield:H30269", "etf_bars:SH512890", "etf_actions:SH512890"} & blocked_ids),
    }
