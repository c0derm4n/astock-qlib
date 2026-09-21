"""逐日可交易池：使用当时的行情、已公告规模和观察期，不回填最新池。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
import universe
from src.utils import symbol_to_code


def historical_universe(bars, nav, meta, *, min_days=None, min_amount=None,
                        min_aum=None, require_aum=True) -> pd.DataFrame:
    min_days = config.UNIVERSE_MIN_LIST_DAYS if min_days is None else min_days
    min_amount = config.UNIVERSE_MIN_AVG_AMOUNT if min_amount is None else min_amount
    min_aum = config.UNIVERSE_MIN_AUM if min_aum is None else min_aum
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(bars["date"].unique()).sort_values()
    nav = nav.copy()
    for col in ("nav_date", "ann_date"):
        if col in nav:
            nav[col] = pd.to_datetime(nav[col], errors="coerce").astype("datetime64[ns]")
    listed = dict(zip(meta.get("symbol", []), pd.to_datetime(meta.get("list_date", []))))
    delisted = dict(zip(meta.get("symbol", []), pd.to_datetime(meta.get("delist_date", pd.Series(pd.NaT, index=meta.index)))))
    rows = []
    for symbol, group in bars.groupby("symbol", sort=True):
        g = group.sort_values("date").set_index("date").reindex(calendar)
        observed = g["close"].notna()
        age = observed.cumsum()
        listing = listed.get(symbol)
        # 两年以上存量ETF豁免截断窗口；近期上市者累计实际观察bar，禁止近似提前放行。
        if pd.notna(listing) and listing < calendar[0] - pd.Timedelta(days=730):
            age = age + min_days
        avg = g["amount"].rolling(20, min_periods=20).mean()
        ns = nav[nav["symbol"] == symbol].copy() if not nav.empty else pd.DataFrame()
        aum = pd.Series(np.nan, index=calendar)
        if not ns.empty:
            ns = ns.dropna(subset=["ann_date", "nav_date", "net_asset"])
            ns = ns[ns["ann_date"] >= ns["nav_date"]]
            ns = ns.sort_values(["ann_date", "nav_date"]).drop_duplicates("ann_date", keep="last")
            if not ns.empty:
                joined = pd.merge_asof(
                    pd.DataFrame({"date": calendar}), ns[["ann_date", "net_asset"]],
                    left_on="date", right_on="ann_date", direction="backward",
                    allow_exact_matches=False,
                )
                aum = pd.Series(joined["net_asset"].to_numpy(), index=calendar)
        size_ok = aum.ge(min_aum) | (aum.isna() & (not require_aum))
        ok = observed & age.ge(min_days) & avg.ge(min_amount) & size_ok
        if pd.notna(delisted.get(symbol)):
            ok &= calendar < delisted[symbol]
        if config.EXCLUDE_QDII and universe.get_asset_class(symbol_to_code(symbol)) == "qdii":
            ok[:] = False
        rows.append(pd.DataFrame({"date": calendar, "symbol": symbol,
                                  "eligible": ok.to_numpy(), "avg_amount": avg.to_numpy(),
                                  "known_aum": aum.to_numpy(), "observed_days": age.to_numpy()}))
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "eligible", "avg_amount", "known_aum", "observed_days"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def universe_from_store(store):
    from src.catalog import read_lifecycle, merge_lifecycle
    meta = merge_lifecycle(store.load_meta(), read_lifecycle())
    return historical_universe(store.load_bars_raw(), store.con.execute("SELECT * FROM nav").df(), meta)


def build_dynamic_universe(store, as_of=None, verbose=True) -> list[str]:
    frame = universe_from_store(store)
    if frame.empty:
        return []
    cutoff = pd.Timestamp(as_of) if as_of else frame["date"].max()
    dates = frame.loc[frame["date"] <= cutoff, "date"]
    if dates.empty:
        return []
    return frame[(frame["date"] == dates.max()) & frame["eligible"]]["symbol"].sort_values().tolist()


def eligibility_intervals(frame) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for symbol, group in frame.groupby("symbol"):
        group = group.sort_values("date")
        blocks = group["eligible"].ne(group["eligible"].shift()).cumsum()
        intervals = []
        for _, block in group[group["eligible"]].groupby(blocks):
            intervals.append((str(block["date"].iloc[0].date()), str(block["date"].iloc[-1].date())))
        if intervals:
            out[str(symbol)] = intervals
    return out
