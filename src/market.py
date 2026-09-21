"""只读市场快照与因果特征；研究、生产模型和决策共用。"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
import universe
from src.datasource import DataStore
from src.temporal import label_end_dates
from src.universe_filter import historical_universe
from src.catalog import read_lifecycle, merge_lifecycle


@dataclass
class Market:
    bars: pd.DataFrame
    eligibility: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    label_ends: pd.Series
    calendar: pd.DatetimeIndex
    fingerprint: str
    audit: dict
    nav: pd.DataFrame | None = None
    meta: pd.DataFrame | None = None

    def wide(self, field):
        return self.bars.pivot(index="date", columns="symbol", values=field).reindex(self.calendar)


def causal_features(bars, horizon=20):
    calendar = pd.DatetimeIndex(bars["date"].unique()).sort_values()
    wide = lambda col: bars.pivot(index="date", columns="symbol", values=col).reindex(calendar)
    close, opened = wide("adj_close"), wide("adj_open")
    returns = close.pct_change(fill_method=None)
    amount, volume = wide("amount"), wide("volume")
    fields = {}
    for days in (5, 20, 60, 120):
        fields[f"momentum_{days}"] = close / close.shift(days) - 1
    for days in (20, 60):
        fields[f"volatility_{days}"] = returns.rolling(days, min_periods=days).std()
        fields[f"mean_deviation_{days}"] = close / close.rolling(days).mean() - 1
    fields["drawdown_60"] = close / close.rolling(60).max() - 1
    fields["range_20"] = ((wide("high") - wide("low")) / wide("close")).rolling(20).mean()
    fields["volume_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    fields["amount_ratio"] = amount.rolling(5).mean() / amount.rolling(20).mean()
    fields["downside_20"] = returns.clip(upper=0).pow(2).rolling(20).mean().pow(.5)
    fields["skip_momentum"] = close.shift(20) / close.shift(120) - 1
    feature = pd.concat({key: value.stack(future_stack=True) for key, value in fields.items()}, axis=1)
    feature.index.names = ["datetime", "instrument"]
    label = (opened.shift(-(horizon + 1)) / opened.shift(-1) - 1).stack(future_stack=True)
    label.index.names = feature.index.names
    label.name = "label"
    ends = label_end_dates(calendar, horizon).reindex(feature.index.get_level_values(0))
    ends.index = feature.index
    return feature.replace([np.inf, -np.inf], np.nan), label, ends


def load_market(path=None, as_of=None, horizon=20) -> Market:
    with DataStore(path, read_only=True) as store:
        bars = store.con.execute(
            "SELECT b.*, a.adj_factor FROM bars_raw b LEFT JOIN adj_factor a "
            "ON b.symbol=a.symbol AND b.date=a.date ORDER BY b.date,b.symbol"
        ).df()
        nav = store.con.execute("SELECT * FROM nav ORDER BY symbol,nav_date,ann_date").df()
        meta = store.load_meta().sort_values("symbol")
        adj_sources = [r[0] for r in store.con.execute("SELECT DISTINCT source FROM adj_factor ORDER BY source").fetchall()]
    lifecycle = read_lifecycle()
    meta = merge_lifecycle(meta, lifecycle)
    if bars.empty:
        raise ValueError("市场数据库无行情")
    bars["date"] = pd.to_datetime(bars["date"])
    if as_of:
        bars = bars[bars["date"] <= pd.Timestamp(as_of)].copy()
    if bars.empty:
        raise ValueError("指定日期之前无行情")
    if bars.duplicated(["date", "symbol"]).any():
        raise ValueError("重复行情主键")
    price_fields = ["open", "high", "low", "close", "adj_factor"]
    if not np.isfinite(bars[price_fields]).all().all() or (bars[price_fields] <= 0).any().any():
        raise ValueError("行情或复权因子缺失/非正，停止研究与决策")
    if (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("行情最高价不符合OHLC关系")
    if (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("行情最低价不符合OHLC关系")
    sources = sorted(bars["source"].dropna().unique().tolist())
    if not sources or not set(sources).issubset({"tushare", "akshare_em_raw", "tencent_raw"}):
        raise ValueError("审计路径仅接受明确不复权日线；旧AKShare混合复权历史须先统一重建")
    for field in ("open", "close"):
        bars["adj_" + field] = bars[field] * bars["adj_factor"]
    bars["vwap"] = bars["amount"] / (bars["volume"] * 100).replace(0, np.nan)
    eligibility = historical_universe(bars, nav, meta)
    feature, labels, ends = causal_features(bars, horizon)
    digest = hashlib.sha256()
    for frame in (bars, nav, meta):
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    calendar = pd.DatetimeIndex(bars["date"].unique()).sort_values()
    audit = {"sources": sources, "bars": len(bars), "symbols": bars["symbol"].nunique(),
             "start": str(calendar[0].date()), "end": str(calendar[-1].date()),
             "universe_scope": "current_curated_list_survivorship_not_audited",
             "return_basis": "adjustment_factor_total_return_proxy", "nav_availability": "ann_date < signal_date"}
    audit["lifecycle_rows"] = len(lifecycle)
    audit["lifecycle_symbols_without_bars"] = sorted(set(lifecycle["symbol"]) - set(bars["symbol"]))
    latest = bars.groupby("symbol")["date"].max()
    excluded = set(meta.loc[pd.to_datetime(meta.get("delist_date", pd.Series(index=meta.index, dtype="datetime64[ns]"))).le(calendar[-1]), "symbol"])
    audit["stale_symbols"] = [s for s, day in latest.items() if day < calendar[-1]
                              and universe.get_asset_class(s[2:]) != "qdii" and s not in excluded]
    audit["adjustment_sources"] = adj_sources
    audit["nav_latest_information_date"] = str(pd.to_datetime(nav["ann_date"]).max().date()) if not nav.empty else None
    return Market(bars, eligibility, feature, labels, ends, calendar, digest.hexdigest(), audit, nav, meta)
