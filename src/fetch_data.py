"""第 1 步：抓取 A股 ETF 数据到 DuckDB 规范化真源（Tushare 主 + AKShare 校验）。

- 日线 OHLCV：Tushare `fund_daily`（不复权，主源；config.PRIMARY_BAR_SOURCE 可切 akshare 兜底）
- 复权因子：Tushare `fund_adj`
- 净值/规模：Tushare `fund_nav`
- 元数据：Tushare `etf_basic`（8000积分不足自动降级 fund_basic/AKShare）
- 血缘：每次拉取写 ingest_log；抽样交叉校验写 validation_log
- 增量：按 DuckDB 已有最新日期续拉；--force 全量重拉；--no-validate 跳过校验

用法:
    python -m src.fetch_data                # 增量 + 校验
    python -m src.fetch_data --force        # 全量重拉
    python -m src.fetch_data --no-validate  # 跳过 AKShare 抽样校验
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd

import config
import universe
from src.datasource import DataStore
from src.datasource import akshare_source as ak_src
from src.datasource import tushare_source as ts_src
from src.utils import to_qlib_symbol


def get_universe() -> list[str]:
    codes = universe.get_codes()
    if config.MAX_STOCKS:
        codes = codes[: config.MAX_STOCKS]
    return codes


def _now() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_id() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")


def _stamp(df: pd.DataFrame, source: str, run_id: str) -> pd.DataFrame:
    """给规范化 df 补上血缘列（source/fetch_time/data_version）。"""
    df = df.copy()
    df["source"] = source
    df["fetch_time"] = _now()
    df["data_version"] = run_id
    return df


def _incremental_start(store: DataStore, symbol: str, force: bool) -> str:
    """增量起点：已有数据则从最后一天(含)续拉以覆盖可能的除权修正；否则全量起始。"""
    if force:
        return config.START_DATE
    mx = store.bar_date_max(symbol)
    return mx.replace("-", "") if mx else config.START_DATE


def _date_range(df: pd.DataFrame, col: str) -> tuple[str | None, str | None]:
    if df is None or df.empty:
        return None, None
    return str(df[col].iloc[0]), str(df[col].iloc[-1])


def ingest(store: DataStore, codes: list[str], run_id: str, force: bool) -> None:
    """拉取 bars(主源) + adj + nav，写入 DuckDB 并记录血缘。"""
    use_ak_bars = config.PRIMARY_BAR_SOURCE == "akshare"
    bar_source_name = "akshare" if use_ak_bars else "tushare"
    end = config.END_DATE
    logs: list[dict] = []
    ok = fail = 0
    skip_ifaces: set[str] = set()   # 积分/权限不足的接口，后续标的直接跳过

    def _do(iface: str, source: str, fn, symbol: str, date_col: str):
        """拉取一个接口并写库/记账；返回行数。积分类错误会标记该接口跳过。"""
        if iface in skip_ifaces:
            logs.append(dict(run_id=run_id, source=source, interface=iface, symbol=symbol,
                             rows=0, date_min=None, date_max=None, fetch_time=_now(),
                             data_version=run_id, status="skip", error="接口积分不足已跳过"))
            return -1
        try:
            df = fn()
            n = 0 if df is None else len(df)
            if df is not None and not df.empty:
                if iface == "fund_daily":
                    store.upsert_bars(_stamp(df, source, run_id))
                elif iface == "fund_adj":
                    store.upsert_adj(_stamp(df, source, run_id))
                elif iface == "fund_nav":
                    store.upsert_nav(_stamp(df, source, run_id))
            d0, d1 = _date_range(df, date_col)
            logs.append(dict(run_id=run_id, source=source, interface=iface, symbol=symbol,
                             rows=n, date_min=d0, date_max=d1, fetch_time=_now(),
                             data_version=run_id, status="ok" if n else "empty", error=""))
            return n
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("权限", "积分", "permission", "没有访问")):
                skip_ifaces.add(iface)
                print(f"  [跳过接口] {iface}：积分/权限不足，后续标的不再尝试（{msg[:60]}）")
            logs.append(dict(run_id=run_id, source=source, interface=iface, symbol=symbol,
                             rows=0, date_min=None, date_max=None, fetch_time=_now(),
                             data_version=run_id, status="fail", error=msg[:300]))
            return -2

    print(f"开始拉取（每只 3 个接口，逐只打印进度）…")
    # 无 token 时跳过仅 Tushare 提供的接口（NAV；若主源也是 tushare 则一并跳过）
    if not config.TUSHARE_TOKEN:
        skip_ifaces.add("fund_nav")
        if not use_ak_bars:
            skip_ifaces.update({"fund_daily", "fund_adj"})
        print("提示：未设置 TUSHARE_TOKEN，已跳过依赖 Tushare 的接口（akshare 主源自带后复权）")
    for i, code in enumerate(codes, 1):
        symbol = to_qlib_symbol(code)
        if symbol is None:
            continue
        start = _incremental_start(store, symbol, force)
        nb = _do("fund_daily", bar_source_name,
                 lambda: (ak_src.fetch_bars(code, start, end) if use_ak_bars
                          else ts_src.fetch_bars(code, start, end)), symbol, "date")
        adj_source = "akshare" if use_ak_bars else "tushare"
        na = _do("fund_adj", adj_source,
                 lambda: (ak_src.fetch_adj(code, start, end) if use_ak_bars
                          else ts_src.fetch_adj(code, start, end)), symbol, "date")
        nn = _do("fund_nav", "tushare", lambda: ts_src.fetch_nav(code, start, end), symbol, "nav_date")
        ok += 1 if nb and nb > 0 else 0
        fail += 1 if nb == -2 else 0

        def _tag(x: int) -> str:
            return "skip" if x == -1 else ("fail" if x == -2 else str(x))
        print(f"[{i}/{len(codes)}] {symbol}  bars={_tag(nb)} adj={_tag(na)} nav={_tag(nn)}", flush=True)
        time.sleep(0.05)

    store.log_ingest(logs)
    print(f"\n拉取完成：日线成功 {ok}，失败 {fail}（明细：python -m src.data_report）")


def ingest_meta(store: DataStore, codes: list[str], run_id: str) -> None:
    """元数据：Tushare etf_basic/fund_basic 优先，异常降级 AKShare。"""
    try:
        meta = ts_src.fetch_meta(codes)
        src = "tushare"
    except Exception:
        meta = ak_src.fetch_meta(codes)
        src = "akshare"
    if meta is not None and not meta.empty:
        store.upsert_meta(meta)
    store.log_ingest([dict(run_id=run_id, source=src, interface="etf_basic",
                           symbol="*", rows=0 if meta is None else len(meta),
                           date_min=None, date_max=None, fetch_time=_now(),
                           data_version=run_id, status="ok" if meta is not None else "fail",
                           error="")])
    print(f"元数据：{0 if meta is None else len(meta)} 条（源={src}）")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def cross_validate(store: DataStore, codes: list[str], run_id: str) -> None:
    """抽样交叉校验：Tushare(主) ↔ AKShare(校验) 的 close/收益 + 复权一致性。"""
    n_sample = int(getattr(config, "VALIDATE_SAMPLE_N", 0))
    if n_sample <= 0:
        return
    rel_tol = config.VALIDATE_REL_TOL
    adj_tol = config.VALIDATE_ADJ_TOL
    rows: list[dict] = []
    passed_cnt = 0
    print("\n开始抽样交叉校验（Tushare ↔ AKShare）…")
    for code in codes:
        symbol = to_qlib_symbol(code)
        if symbol is None:
            continue
        # 主源已入库的后复权/原价
        raw = store.load_bars_raw([symbol])
        hfq = store.load_hfq_bars([symbol]).get(symbol)
        if raw is None or raw.empty:
            continue
        try:
            ak_raw = ak_src.fetch_bars(code, config.START_DATE, config.END_DATE)
        except Exception as e:
            rows.append(dict(run_id=run_id, symbol=symbol, check_type="ohlcv_cross",
                             n_checked=0, n_mismatch=0, max_rel_diff=None, passed=False,
                             run_time=_now(), note=f"akshare 拉取失败:{str(e)[:120]}"))
            continue
        if ak_raw is None or ak_raw.empty:
            continue

        # ---- ohlcv_cross：收盘价抽样比对 ----
        m = raw[["date", "close"]].merge(ak_raw[["date", "close"]], on="date",
                                         suffixes=("_ts", "_ak"))
        if not m.empty:
            idx = list(range(len(m)))
            random.shuffle(idx)
            idx = idx[: min(n_sample, len(m))]
            diffs = [_rel(m["close_ts"].iloc[k], m["close_ak"].iloc[k]) for k in idx]
            n_mis = int(sum(d > rel_tol for d in diffs))
            mx = float(max(diffs)) if diffs else 0.0
            ok = n_mis == 0
            passed_cnt += int(ok)
            rows.append(dict(run_id=run_id, symbol=symbol, check_type="ohlcv_cross",
                             n_checked=len(idx), n_mismatch=n_mis, max_rel_diff=round(mx, 6),
                             passed=ok, run_time=_now(),
                             note="" if ok else f"{n_mis}个采样点close相对误差>{rel_tol}"))
            if not ok:
                print(f"  [告警] {symbol} OHLCV 校验 {n_mis}/{len(idx)} 超阈(max={mx:.4f})，可人工补数")

        # ---- adj_consistency：后复权日收益一致性（vs AKShare hfq）----
        try:
            import akshare as ak
            from src.utils import to_ak_code
            df_hfq = ak.fund_etf_hist_em(symbol=to_ak_code(code), period="daily",
                                         start_date=config.START_DATE,
                                         end_date=config.END_DATE, adjust="hfq")
        except Exception:
            df_hfq = None
        if hfq is not None and df_hfq is not None and not df_hfq.empty:
            df_hfq = df_hfq.rename(columns={"日期": "date", "收盘": "close_akh"})
            df_hfq["date"] = pd.to_datetime(df_hfq["date"]).dt.strftime("%Y-%m-%d")
            ts_ret = hfq[["date", "close"]].copy()
            ts_ret["ret_ts"] = ts_ret["close"].pct_change()
            j = ts_ret.merge(df_hfq[["date", "close_akh"]], on="date")
            j["ret_ak"] = j["close_akh"].astype(float).pct_change()
            j = j.dropna()
            if not j.empty:
                idx = list(range(len(j)))
                random.shuffle(idx)
                idx = idx[: min(n_sample, len(j))]
                diffs = [abs(j["ret_ts"].iloc[k] - j["ret_ak"].iloc[k]) for k in idx]
                n_mis = int(sum(d > adj_tol for d in diffs))
                mx = float(max(diffs)) if diffs else 0.0
                ok = n_mis == 0
                rows.append(dict(run_id=run_id, symbol=symbol, check_type="adj_consistency",
                                 n_checked=len(idx), n_mismatch=n_mis, max_rel_diff=round(mx, 6),
                                 passed=ok, run_time=_now(),
                                 note="" if ok else f"复权日收益{n_mis}点差异>{adj_tol}"))
        time.sleep(0.2)

    if rows:
        store.log_validation(rows)
    print(f"交叉校验完成：{passed_cnt} 只 OHLCV 通过（明细见 validation_log）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="全量重拉(忽略增量)")
    parser.add_argument("--no-validate", action="store_true", help="跳过 AKShare 抽样校验")
    args = parser.parse_args()

    codes = get_universe()
    print(f"ETF池(宽基+行业主题+防守)：共 {len(codes)} 只；日线主源={config.PRIMARY_BAR_SOURCE}")
    run_id = _run_id()
    print(f"run_id={run_id}")

    with DataStore() as store:
        ingest(store, codes, run_id, args.force)
        ingest_meta(store, codes, run_id)
        if not args.no_validate:
            cross_validate(store, codes, run_id)

    print(f"\n完成。DuckDB：{config.DUCKDB_PATH}")


if __name__ == "__main__":
    main()
