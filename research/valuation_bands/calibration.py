"""年度校准: 历史低点估值中位数、分位退出线, 以及冻结快照。

严格按 README 第4节: 只用信号年 y 之前的 5 个完整自然年; 并列最低价取最早日期;
分位采用线性插值(Type 7); 低点当天估值不做插值; 阈值必须严格有序。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .data import ResearchData, sha256_json, session_ts


class CalibrationError(RuntimeError):
    """校准不可用; ``reason`` 给出机器可读原因。"""

    def __init__(self, reason: str, detail: str, evidence: Optional[Mapping[str, object]] = None):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.evidence = dict(evidence or {})


def quantile_type7(values: Sequence[float], q: float) -> float:
    """线性插值分位(numpy/pandas 默认 linear, 即 Type 7)。"""
    array = np.asarray([float(v) for v in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("空样本无法计算分位")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"分位参数越界: {q}")
    return float(np.quantile(array, q, method="linear"))


@dataclass
class CalibrationSnapshot:
    strategy: str
    index_id: str
    metric: str
    year: int
    lookback_years: List[int]
    q: float
    lows: List[Dict[str, object]]
    median_low_value: float
    l1: float
    l2: float
    l3: float
    exit_level: float
    sample_count: int
    price_coverage: Dict[int, float]
    valuation_coverage: Dict[int, float]
    snapshot_hash: str = ""
    created_at: str = ""
    params_hash: str = ""

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "strategy": self.strategy,
            "index_id": self.index_id,
            "metric": self.metric,
            "year": self.year,
            "lookback_years": ";".join(str(y) for y in self.lookback_years),
            "q": self.q,
            "median_low_value": self.median_low_value,
            "L1": self.l1,
            "L2": self.l2,
            "L3": self.l3,
            "exit_level": self.exit_level,
            "sample_count": self.sample_count,
            "snapshot_hash": self.snapshot_hash,
            "params_hash": self.params_hash,
            "created_at": self.created_at,
        }
        for entry in self.lows:
            year = entry["year"]
            row[f"low_{year}_date"] = entry["trade_date"]
            row[f"low_{year}_close"] = entry["close"]
            row[f"low_{year}_value"] = entry["value"]
            row[f"price_cov_{year}"] = self.price_coverage.get(year)
            row[f"val_cov_{year}"] = self.valuation_coverage.get(year)
        return row

    def thresholds(self) -> Dict[str, float]:
        return {"L1": self.l1, "L2": self.l2, "L3": self.l3, "S": self.exit_level}

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "index_id": self.index_id,
            "metric": self.metric,
            "year": self.year,
            "lookback_years": self.lookback_years,
            "q": self.q,
            "lows": self.lows,
            "median_low_value": self.median_low_value,
            "thresholds": self.thresholds(),
            "sample_count": self.sample_count,
            "price_coverage": self.price_coverage,
            "valuation_coverage": self.valuation_coverage,
            "snapshot_hash": self.snapshot_hash,
            "params_hash": self.params_hash,
            "created_at": self.created_at,
        }


def _year_price_slice(price: pd.DataFrame, year: int, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    sessions = calendar[calendar.year == year]
    frame = price[price["trade_date"].isin(sessions)].copy()
    return frame.sort_values("trade_date")


def annual_low(price: pd.DataFrame, valuation: pd.DataFrame, year: int, calendar: pd.DatetimeIndex,
               price_cov_min: float, val_cov_min: float) -> Dict[str, object]:
    """返回该年价格最低日及其同日估值, 附覆盖率证据。"""
    sessions = calendar[calendar.year == year]
    if len(sessions) == 0:
        raise CalibrationError("annual_calendar_empty", f"{year} 年没有交易日")
    frame = _year_price_slice(price, year, calendar)
    if frame["trade_date"].duplicated().any() or valuation[valuation["trade_date"].isin(sessions)]["trade_date"].duplicated().any():
        raise CalibrationError("duplicate_observations", f"{year} 年存在重复日观测")
    frame = frame[np.isfinite(frame["close"].astype(float)) & (frame["close"].astype(float) > 0)]
    price_cov = len(frame) / len(sessions)
    if price_cov < price_cov_min:
        raise CalibrationError(
            "annual_price_incomplete",
            f"{year} 年价格覆盖 {price_cov:.2%} < {price_cov_min:.0%}, 可能遗漏真正最低日",
            {"year": year, "sessions": int(len(sessions)), "observed": int(len(frame))},
        )
    val = valuation[valuation["trade_date"].isin(sessions)]
    val = val[np.isfinite(val["value"].astype(float)) & (val["value"].astype(float) > 0)]
    val_cov = len(val) / len(sessions)
    if val_cov < val_cov_min:
        raise CalibrationError(
            "annual_valuation_incomplete",
            f"{year} 年估值覆盖 {val_cov:.2%} < {val_cov_min:.0%}",
            {"year": year, "sessions": int(len(sessions)), "observed": int(len(val))},
        )
    frame = frame[np.isfinite(frame["close"].astype(float)) & (frame["close"].astype(float) > 0)]
    if frame.empty:
        raise CalibrationError("annual_price_invalid", f"{year} 年无有效收盘价")
    lowest = float(frame["close"].astype(float).min())
    low_rows = frame[np.isclose(frame["close"].astype(float), lowest, rtol=0, atol=1e-9)]
    low_date = pd.Timestamp(low_rows["trade_date"].min())  # 并列取最早
    hit = val[val["trade_date"] == low_date]
    if len(hit) != 1:
        raise CalibrationError(
            "annual_low_valuation_missing",
            f"{year} 年最低日 {low_date.date()} 无唯一同日估值, 不做插值",
            {"year": year, "date": str(low_date.date())},
        )
    value = float(hit.iloc[0]["value"])
    if not np.isfinite(value) or value <= 0:
        raise CalibrationError("annual_low_valuation_invalid", f"{year} 年最低日估值非正或非有限")
    return {
        "year": year,
        "trade_date": str(low_date.date()),
        "close": lowest,
        "value": value,
        "price_coverage": round(price_cov, 6),
        "valuation_coverage": round(val_cov, 6),
        "tie_count": int(len(low_rows)),
    }


def calibrate_year(
    data: ResearchData,
    index_id: str,
    metric: str,
    year: int,
    params: Mapping[str, object],
    strategy: str = "A",
    q: float = 0.85,
    fixed_exit_level: Optional[float] = None,
    snapshot_ts: Optional[pd.Timestamp] = None,
) -> CalibrationSnapshot:
    """为信号年 ``year`` 生成冻结校准快照。"""
    lookback = int(params.get("lookback_years", 5))
    price_cov_min = float(params.get("price_coverage_min", 1.0))
    val_cov_min = float(params.get("valuation_coverage_min", 0.95))
    min_obs = int(params.get("min_valuation_observations", 1000))
    years = list(range(year - lookback, year))
    price = data.index_price(index_id)
    valuation = data.index_valuation(index_id, metric)
    if price.empty:
        raise CalibrationError("calibration_data_blocked", f"缺少指数 {index_id} 价格序列")
    if valuation.empty:
        raise CalibrationError(
            "calibration_data_blocked",
            f"缺少指数 {index_id} 的 {metric} 日频估值序列",
            {"index_id": index_id, "metric": metric},
        )

    stamp = snapshot_ts if snapshot_ts is not None else session_ts(pd.Timestamp(f"{year}-01-01"))
    for name, observations in (("valuation", valuation), ("price", price)):
        historical = observations[observations["trade_date"].dt.year.isin(years)]
        availability = historical["effective_available_at"]
        if availability.isna().any() or (availability > stamp).any():
            raise CalibrationError(f"lookahead_{name}", f"校准包含评估时尚不可得的{name}记录")

    lows: List[Dict[str, object]] = []
    price_cov: Dict[int, float] = {}
    val_cov: Dict[int, float] = {}
    for k in years:
        entry = annual_low(price, valuation, k, data.calendar, price_cov_min, val_cov_min)
        lows.append(entry)
        price_cov[k] = entry["price_coverage"]
        val_cov[k] = entry["valuation_coverage"]

    window = valuation[valuation["trade_date"].dt.year.isin(years)].copy()
    window = window[np.isfinite(window["value"].astype(float)) & (window["value"].astype(float) > 0)]
    window = window.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
    sample_count = int(len(window))
    if sample_count < min_obs:
        raise CalibrationError(
            "sample_count_insufficient",
            f"{years[0]}-{years[-1]} 有效估值观测 {sample_count} 条 < {min_obs} 条",
            {"sample_count": sample_count},
        )

    stamp = snapshot_ts if snapshot_ts is not None else session_ts(pd.Timestamp(f"{year}-01-01"))
    if window["effective_available_at"].max() > stamp:
        raise CalibrationError(
            "lookahead_valuation",
            f"校准窗口内存在 {stamp} 之后才可得的估值记录",
            {"max_available_at": str(window["effective_available_at"].max())},
        )

    values = [entry["value"] for entry in lows]
    median_low = float(np.median(np.asarray(values, dtype=float)))
    upper = float(params.get("band_upper_mult", 1.1))
    mid = float(params.get("band_mid_mult", 1.0))
    low = float(params.get("band_low_mult", 0.9))
    l1, l2, l3 = upper * median_low, mid * median_low, low * median_low
    exit_level = float(fixed_exit_level) if fixed_exit_level is not None else quantile_type7(window["value"].tolist(), q)

    if not (0 < l3 < l2 < l1 < exit_level):
        raise CalibrationError(
            "invalid_threshold_order",
            f"阈值顺序异常: L3={l3:.6f} L2={l2:.6f} L1={l1:.6f} S={exit_level:.6f}",
            {"thresholds": {"L1": l1, "L2": l2, "L3": l3, "S": exit_level}},
        )

    payload = {
        "strategy": strategy,
        "index_id": index_id,
        "metric": metric,
        "year": year,
        "lookback_years": years,
        "q": q,
        "lows": lows,
        "sample_count": sample_count,
        "thresholds": {"L1": l1, "L2": l2, "L3": l3, "S": exit_level},
    }
    params_hash = sha256_json(
        {k: params.get(k) for k in ("lookback_years", "band_upper_mult", "band_mid_mult", "band_low_mult",
                                    "quantile_method", "valuation_coverage_min", "price_coverage_min",
                                    "min_valuation_observations")}
    )
    return CalibrationSnapshot(
        strategy=strategy,
        index_id=index_id,
        metric=metric,
        year=year,
        lookback_years=years,
        q=q,
        lows=lows,
        median_low_value=median_low,
        l1=l1,
        l2=l2,
        l3=l3,
        exit_level=exit_level,
        sample_count=sample_count,
        price_coverage=price_cov,
        valuation_coverage=val_cov,
        snapshot_hash=sha256_json(payload),
        created_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(timespec="seconds"),
        params_hash=params_hash,
    )


class CalibrationBook:
    """按年缓存校准快照; 年内固定, 不因结果调整, 不被后续数据修订覆盖。"""

    def __init__(self, data: ResearchData, params: Mapping[str, object], index_id: str, metric: str,
                 strategy: str, q: float, fixed_exit_level: Optional[float] = None):
        self.data = data
        self.params = dict(params)
        self.index_id = index_id
        self.metric = metric
        self.strategy = strategy
        self.q = q
        self.fixed_exit_level = fixed_exit_level
        self.snapshots: Dict[int, CalibrationSnapshot] = {}
        self.failures: Dict[int, Dict[str, object]] = {}

    def get(self, year: int, snapshot_ts: Optional[pd.Timestamp] = None) -> Optional[CalibrationSnapshot]:
        """取年度快照; 已生成则冻结复用, 失败则保留原因并在后续周度评估重试。"""
        if year in self.snapshots:
            return self.snapshots[year]
        try:
            snapshot = calibrate_year(
                self.data, self.index_id, self.metric, year, self.params,
                strategy=self.strategy, q=self.q, fixed_exit_level=self.fixed_exit_level,
                snapshot_ts=snapshot_ts,
            )
        except CalibrationError as error:
            self.failures[year] = {"reason": error.reason, "detail": error.detail, "evidence": error.evidence}
            return None
        self.snapshots[year] = snapshot
        self.failures.pop(year, None)
        return snapshot

    def rows(self) -> List[Dict[str, object]]:
        return [self.snapshots[y].to_row() for y in sorted(self.snapshots)]
