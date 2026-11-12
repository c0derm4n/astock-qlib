"""公开免费数据获取与规范化(只写本研究目录)。

只用无需新增费用、无需登录/绕过访问控制的公开端点:
- 中证指数官网公开接口 ``index-perf``(沪深300价格指数日线)
- 国证指数官网公开接口 ``getIndexDailyDataWithDataFormat``(创业板指日线)
- 中证指数官网公开估值文件 ``indicator.xls``(仅滚动20个交易日)
- 东方财富ETF不复权日线(经 akshare ``fund_etf_hist_em``)
- 新浪公司行动 ``hfq.js``(经 ``src.datasource.free_source``)
- 项目已有的 ``data/formal/market.duckdb`` 原始日线(只读)
每次运行都新增 ``raw/<provider>/<fetched_at>/`` 与新的 ``snapshots/<version>/``, 不覆盖历史。
"""
from __future__ import annotations

import hashlib
import io
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .data import build_calendar, sha256_file

UA = {"User-Agent": "Mozilla/5.0 (research; valuation_bands)", "Accept": "application/json, text/plain, */*"}
CSI_INDICATOR_URL = ("https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/"
                     "autofile/indicator/{code}indicator.xls")


def _session(evidence_dir: Optional[Path] = None) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        def archive(response, *args, **kwargs):
            digest = hashlib.sha256(response.content).hexdigest()
            (evidence_dir / f"{digest}.bin").write_bytes(response.content)
            (evidence_dir / f"{digest}.json").write_text(json.dumps({
                "url": response.url, "http_status": response.status_code,
                "sha256": digest, "fetched_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        session.hooks["response"].append(archive)
    return session


def _stamp() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%dT%H%M%S")


def _next_session_19(calendar: pd.DatetimeIndex, dates: pd.Series) -> pd.Series:
    position = {session: pos for pos, session in enumerate(calendar)}
    cache: Dict[pd.Timestamp, pd.Timestamp] = {}
    out = []
    for date in pd.to_datetime(dates):
        date = pd.Timestamp(date).normalize()
        if date not in cache:
            pos = position.get(date)
            if pos is None:
                later = calendar[calendar >= date]
                pos = position[later[0]] if len(later) else len(calendar) - 1
            if pos + 1 < len(calendar):
                nxt = calendar[pos + 1]
            else:
                # 末尾观测也必须延迟到真正的下一交易日，不得压回当日。
                extension = build_calendar(date + pd.Timedelta(days=1), date + pd.Timedelta(days=30))
                nxt = extension[0]
            cache[date] = pd.Timestamp.combine(nxt.date(), datetime.min.time()).replace(hour=19)
        out.append(cache[date])
    return pd.Series(out, index=dates.index)


def _get_json(session: requests.Session, url: str, params: Dict[str, object], tries: int = 3,
              backoff: float = 2.0) -> Tuple[Optional[int], Optional[dict], Optional[str]]:
    """带退避的公开接口请求; 返回 (status, payload, error)。"""
    last_error = None
    for attempt in range(tries):
        try:
            response = session.get(url, params=params, headers=UA, timeout=30)
            if response.status_code == 200:
                try:
                    return response.status_code, response.json(), None
                except Exception as error:
                    last_error = f"non_json:{type(error).__name__}"
            else:
                last_error = f"http_{response.status_code}"
        except Exception as error:
            last_error = f"{type(error).__name__}:{error}"
        time.sleep(backoff * (attempt + 1))
    return None, None, last_error


def _record_probe(probes: List[Dict[str, object]], series: str, provider: str, endpoint: str,
                  status, rows: Optional[int], first=None, last=None, note: str = ""):
    probes.append({
        "series": series,
        "provider": provider,
        "endpoint": endpoint,
        "probed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status if not hasattr(status, "value") else status.value,
        "rows": rows,
        "first": str(first) if first is not None else None,
        "last": str(last) if last is not None else None,
        "note": note,
    })


# ------------------------------------------------------------------ 中证指数价格
def fetch_csindex_price(session: requests.Session, code: str, start: str, end: str,
                        probes: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    failed_years: List[Tuple[int, str]] = []
    start_year = int(start[:4])
    end_year = int(end[:4])
    for year in range(start_year, end_year + 1):
        params = {"indexCode": code, "startDate": f"{year}0101", "endDate": f"{year}1231"}
        if year == start_year:
            params["startDate"] = start.replace("-", "")
        if year == end_year:
            params["endDate"] = end.replace("-", "")
        try:
            status, payload, error = _get_json(
                session, "https://www.csindex.com.cn/csindex-home/perf/index-perf", params)
            if payload is None:
                failed_years.append((year, error))
                _record_probe(probes, f"index_price:{code}", "csindex", "index-perf", "error", 0,
                              note=f"year={year} {error}")
                continue
            data = payload.get("data") or []
            for item in data:
                rows.append({
                    "index_id": code,
                    "trade_date": pd.Timestamp(item["tradeDate"]),
                    "close": float(item["close"]),
                    "price_type": "official_index_close",
                    "source_id": "csindex_index_perf",
                })
            _record_probe(probes, f"index_price:{code}", "csindex", "index-perf", status,
                          len(data), data[0]["tradeDate"] if data else None,
                          data[-1]["tradeDate"] if data else None, f"year={year}")
        except Exception as error:
            _record_probe(probes, f"index_price:{code}", "csindex", "index-perf", "error", 0,
                          note=f"year={year} {type(error).__name__}: {error}")
        time.sleep(0.8)
    if failed_years:
        # 任一年份缺失都不能拼出完整序列, 交给调用方整体回退
        _record_probe(probes, f"index_price:{code}", "csindex", "index-perf", "incomplete_series",
                      0, note=f"failed_years={failed_years}")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)


def fetch_index_eastmoney(session: requests.Session, code: str, start: str, end: str,
                          probes: List[Dict[str, object]]) -> pd.DataFrame:
    """备用源: 东财指数日线(整条序列同源, 不与官方序列混接)。"""
    try:
        import akshare as ak

        frame = ak.index_zh_a_hist(symbol=code, period="daily",
                                   start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        frame = frame.rename(columns={"日期": "trade_date", "收盘": "close"})
        out = pd.DataFrame({
            "index_id": code,
            "trade_date": pd.to_datetime(frame["trade_date"]),
            "close": frame["close"].astype(float),
            "price_type": "price_index_close",
            "source_id": "eastmoney_index_zh_a_hist",
        })
        _record_probe(probes, f"index_price:{code}", "eastmoney", "index_zh_a_hist", "ok", len(out),
                      out["trade_date"].min() if not out.empty else None,
                      out["trade_date"].max() if not out.empty else None,
                      "官方源不可用时的整条备用序列")
        return out.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
    except Exception as error:
        _record_probe(probes, f"index_price:{code}", "eastmoney", "index_zh_a_hist", "error", 0,
                      note=f"{type(error).__name__}: {error}")
        return pd.DataFrame()


# ------------------------------------------------------------------ 国证指数价格
def fetch_cnindex_price(session: requests.Session, code: str, start: str, end: str,
                        probes: List[Dict[str, object]]) -> pd.DataFrame:
    try:
        response = session.get(
            "https://www.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat",
            params={"indexCode": code, "startDate": start, "endDate": end, "frequency": "day"},
            headers=UA, timeout=40,
        )
        payload = response.json()
        block = payload.get("data") or {}
        items = block.get("item") or []
        data = block.get("data") or []
        close_pos = items.index("close") if "close" in items else 5
        date_pos = items.index("timestamp") if "timestamp" in items else 0
        rows = []
        for record in data:
            rows.append({
                "index_id": code,
                "trade_date": pd.Timestamp(record[date_pos]),
                "close": float(record[close_pos]),
                "price_type": "official_index_close",
                "source_id": "cnindex_index_daily",
            })
        frame = pd.DataFrame(rows)
        _record_probe(probes, f"index_price:{code}", "cnindex", "getIndexDailyDataWithDataFormat",
                      response.status_code, len(frame),
                      frame["trade_date"].min() if not frame.empty else None,
                      frame["trade_date"].max() if not frame.empty else None)
        if frame.empty:
            return frame
        return frame.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
    except Exception as error:
        _record_probe(probes, f"index_price:{code}", "cnindex", "getIndexDailyDataWithDataFormat",
                      "error", 0, note=f"{type(error).__name__}: {error}")
        return pd.DataFrame()


# ------------------------------------------------------------------ 中证估值文件
def fetch_csindex_indicator(session: requests.Session, code: str, probes: List[Dict[str, object]]) -> pd.DataFrame:
    url = CSI_INDICATOR_URL.format(code=code)
    try:
        response = session.get(url, headers=UA, timeout=30)
        _record_probe(probes, f"index_valuation:{code}:PE_TTM", "csindex", "indicator.xls",
                      response.status_code, None, note=f"bytes={len(response.content)}")
        if response.status_code != 200 or len(response.content) < 500:
            return pd.DataFrame()
        frame = pd.read_excel(io.BytesIO(response.content))
        columns = list(frame.columns)
        frame.columns = ["trade_date", "index_code", "name_cn_full", "name_cn", "name_en_full",
                         "name_en", "pe_total", "pe_calc", "dy_total", "dy_calc"][: len(columns)]
        frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d")
        _record_probe(probes, f"index_valuation:{code}:PE_TTM", "csindex", "indicator.xls",
                      response.status_code, len(frame), frame["trade_date"].min(), frame["trade_date"].max(),
                      note="官方估值文件仅滚动20个交易日, 不足以构成日频历史")
        return frame
    except Exception as error:
        _record_probe(probes, f"index_valuation:{code}:PE_TTM", "csindex", "indicator.xls", "error", 0,
                      note=f"{type(error).__name__}: {error}")
        return pd.DataFrame()


# ------------------------------------------------------------------ ETF 行情
def fetch_etf_from_project_db(symbol: str, probes: List[Dict[str, object]]) -> pd.DataFrame:
    import duckdb

    db = Path(__file__).resolve().parents[2] / "data" / "formal" / "market.duckdb"
    if not db.exists():
        _record_probe(probes, f"etf_bars:{symbol}", "project_duckdb", str(db), "missing", 0)
        return pd.DataFrame()
    con = duckdb.connect(str(db), read_only=True)
    try:
        frame = con.execute(
            "select date, open, high, low, close, volume, amount from bars_raw where symbol = ? order by date",
            [symbol],
        ).df()
    finally:
        con.close()
    if frame.empty:
        _record_probe(probes, f"etf_bars:{symbol}", "project_duckdb", str(db), "empty", 0)
        return pd.DataFrame()
    out = pd.DataFrame({
        "symbol": symbol,
        "trade_date": pd.to_datetime(frame["date"]),
        "open": frame["open"].astype(float),
        "high": frame["high"].astype(float),
        "low": frame["low"].astype(float),
        "close": frame["close"].astype(float),
        "volume_shares": frame["volume"].astype(float) * 100.0,  # 项目库量为手
        "amount_cny": frame["amount"].astype(float),
        "source_id": "project_formal_duckdb",
    })
    _record_probe(probes, f"etf_bars:{symbol}", "project_duckdb", "bars_raw", "ok", len(out),
                  out["trade_date"].min(), out["trade_date"].max(), "只读复用, 量为手已转份")
    return out


def fetch_etf_eastmoney(symbol: str, probes: List[Dict[str, object]]) -> pd.DataFrame:
    try:
        import akshare as ak

        code = symbol[2:]
        frame = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20000101",
                                    end_date="20500101", adjust="")
        if frame is None or frame.empty:
            _record_probe(probes, f"etf_bars:{symbol}", "eastmoney", "fund_etf_hist_em", "empty", 0)
            return pd.DataFrame()
        rename = {"日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
                  "收盘": "close", "成交量": "volume", "成交额": "amount"}
        frame = frame.rename(columns=rename)
        out = pd.DataFrame({
            "symbol": symbol,
            "trade_date": pd.to_datetime(frame["trade_date"]),
            "open": frame["open"].astype(float),
            "high": frame["high"].astype(float),
            "low": frame["low"].astype(float),
            "close": frame["close"].astype(float),
            "volume_shares": frame["volume"].astype(float) * 100.0,
            "amount_cny": frame["amount"].astype(float),
            "source_id": "eastmoney_fund_etf_hist_em",
        })
        out = out[out["trade_date"] <= pd.Timestamp("2026-11-06")]
        _record_probe(probes, f"etf_bars:{symbol}", "eastmoney", "fund_etf_hist_em", "ok", len(out),
                      out["trade_date"].min(), out["trade_date"].max(), "不复权原价, 量为手已转份")
        return out.sort_values("trade_date").reset_index(drop=True)
    except Exception as error:
        _record_probe(probes, f"etf_bars:{symbol}", "eastmoney", "fund_etf_hist_em", "error", 0,
                      note=f"{type(error).__name__}: {error}")
        return pd.DataFrame()


# ------------------------------------------------------------------ 公司行动
def fetch_actions(symbol: str, calendar: pd.DatetimeIndex, probes: List[Dict[str, object]]) -> pd.DataFrame:
    try:
        from src.datasource.free_source import fetch_actions as project_fetch_actions

        frame = project_fetch_actions(symbol)
        if frame is None or frame.empty:
            _record_probe(probes, f"etf_actions:{symbol}", "sina", "hfq.js", "empty", 0)
            return pd.DataFrame()
        frame = frame.sort_values("date").reset_index(drop=True)
        rows = []
        previous_cash = 0.0
        previous_factor = 1.0
        for index, row in frame.iterrows():
            ex_date = pd.Timestamp(row["date"]).normalize()
            cash_diff = float(row["cumulative_cash"]) - previous_cash
            factor = float(row["share_factor"])
            share_factor = factor / previous_factor if previous_factor else factor
            previous_cash = float(row["cumulative_cash"])
            previous_factor = factor
            position = calendar.searchsorted(ex_date, side="right") - 1
            record_date = calendar[max(0, position - 1)]
            pay_position = min(len(calendar) - 1, position + 5)
            payment_date = calendar[pay_position]
            if cash_diff > 1e-9:
                rows.append({
                    "event_id": f"{symbol}-D{index:03d}",
                    "symbol": symbol,
                    "type": "dividend",
                    "announced_at": "",  # 数据源没有公告日期，不能伪造提前12天已知
                    "record_date": str(pd.Timestamp(record_date).date()),
                    "ex_date": str(ex_date.date()),
                    "payment_date": str(pd.Timestamp(payment_date).date()),
                    "cash_per_share": round(cash_diff, 8),
                    "share_factor": 1.0,
                    "evidence_sha256": hashlib.sha256(
                        f"{symbol}|{ex_date.date()}|{cash_diff:.8f}".encode()).hexdigest(),
                })
            if abs(share_factor - 1.0) > 1e-9:
                rows.append({
                    "event_id": f"{symbol}-S{index:03d}",
                    "symbol": symbol,
                    "type": "split",
                    "announced_at": "",
                    "record_date": str(pd.Timestamp(record_date).date()),
                    "ex_date": str(ex_date.date()),
                    "payment_date": "",
                    "cash_per_share": 0.0,
                    "share_factor": round(share_factor, 8),
                    "evidence_sha256": hashlib.sha256(
                        f"{symbol}|{ex_date.date()}|split{share_factor:.8f}".encode()).hexdigest(),
                })
        out = pd.DataFrame(rows)
        _record_probe(probes, f"etf_actions:{symbol}", "sina", "hfq.js", "ok", len(out),
                      out["ex_date"].min() if not out.empty else None,
                      out["ex_date"].max() if not out.empty else None,
                      "登记日/支付日为研究假设(前一日/除息后5个交易日), 已在manifest披露")
        return out
    except Exception as error:
        _record_probe(probes, f"etf_actions:{symbol}", "sina", "hfq.js", "error", 0,
                      note=f"{type(error).__name__}: {error}")
        return pd.DataFrame()


# ------------------------------------------------------------------ 组装快照
def build_real_snapshot(data_root: Path, requested_end: Optional[str] = None) -> Path:
    data_root = Path(data_root)
    requested_end = requested_end or "2026-11-06"
    stamp = _stamp()
    raw_dir = data_root / "raw" / "fetch" / stamp
    raw_dir.mkdir(parents=True, exist_ok=True)
    probes: List[Dict[str, object]] = []
    session = _session(raw_dir / "http")
    socket.setdefaulttimeout(30)

    calendar = build_calendar("2010-01-01", requested_end)

    prices = []
    frame = fetch_csindex_price(session, "000300", "2010-01-04", requested_end, probes)
    if frame.empty:
        frame = fetch_index_eastmoney(session, "000300", "2010-01-04", requested_end, probes)
    if not frame.empty:
        prices.append(frame)
    frame = fetch_cnindex_price(session, "399006", "2010-06-01", requested_end, probes)
    if not frame.empty:
        prices.append(frame)
    index_price = pd.concat(prices, ignore_index=True) if prices else pd.DataFrame()
    if not index_price.empty:
        index_price["effective_available_at"] = _next_session_19(calendar, index_price["trade_date"])
        index_price["availability_basis"] = "assumed_next_session_19"
        index_price["raw_sha256"] = index_price["source_id"]
        index_price = index_price.drop_duplicates(["index_id", "trade_date"]).sort_values(
            ["index_id", "trade_date"]).reset_index(drop=True)

    # 中证官方估值文件(仅20个交易日) - 作为可得性证据与诊断参考, 不作为日频历史
    indicator_300 = fetch_csindex_indicator(session, "000300", probes)
    indicator_h30269 = fetch_csindex_indicator(session, "H30269", probes)
    valuation_rows = []
    for indicator, code in ((indicator_300, "000300"), (indicator_h30269, "H30269")):
        if indicator.empty:
            continue
        for metric, column, basis in (
            ("PE_CSINDEX_PE1", "pe_total", "中证披露市盈率1(总股本口径, 未确认TTM)"),
            ("PE_CSINDEX_PE2", "pe_calc", "中证披露市盈率2(计算用股本口径, 未确认TTM)"),
        ):
            if column not in indicator:
                continue
            for row in indicator.itertuples(index=False):
                value = getattr(row, column)
                if value is None or not np.isfinite(float(value)) or float(value) <= 0:
                    continue
                valuation_rows.append({
                    "index_id": code, "trade_date": row.trade_date, "metric": metric,
                    "value": float(value), "aggregation_method": "中证指数官方披露(整体法)",
                    "financial_basis": basis, "source_id": "csindex_indicator_xls",
                    "source_published_at": None, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "revision_id": "rolling20",
                })
    index_valuation = pd.DataFrame(valuation_rows)
    if not index_valuation.empty:
        index_valuation["effective_available_at"] = _next_session_19(calendar, index_valuation["trade_date"])
        index_valuation["availability_basis"] = "assumed_next_session_19"
        index_valuation["raw_sha256"] = "csindex_indicator_xls"
        _record_probe(probes, "index_valuation:000300:PE_TTM", "csindex", "indicator.xls",
                      "insufficient_history", len(index_valuation),
                      index_valuation["trade_date"].min(), index_valuation["trade_date"].max(),
                      "仅滚动20个交易日, 无法覆盖2011-2015校准窗口")

    dividend_rows = []
    if not indicator_h30269.empty and "dy_total" in indicator_h30269:
        for row in indicator_h30269.itertuples(index=False):
            value = getattr(row, "dy_total")
            if value is None or not np.isfinite(float(value)) or float(value) <= 0:
                continue
            dividend_rows.append({
                "index_id": "H30269", "trade_date": row.trade_date,
                "yield_ttm": float(value) / 100.0,
                "source_id": "csindex_indicator_xls",
                "aggregation_method": "中证披露股息率1(总股本口径, 非TTM确认)",
            })
    dividend_yield = pd.DataFrame(dividend_rows)
    if not dividend_yield.empty:
        dividend_yield["effective_available_at"] = _next_session_19(calendar, dividend_yield["trade_date"])
        dividend_yield["availability_basis"] = "assumed_next_session_19"

    etf_frames = []
    for symbol in ("SH510300", "SZ159915"):
        frame = fetch_etf_from_project_db(symbol, probes)
        if not frame.empty:
            etf_frames.append(frame)
    frame = fetch_etf_eastmoney("SH512890", probes)
    if not frame.empty:
        etf_frames.append(frame)
    etf_bars = pd.concat(etf_frames, ignore_index=True) if etf_frames else pd.DataFrame()
    if not etf_bars.empty:
        etf_bars["available_at"] = etf_bars["trade_date"] + pd.Timedelta(hours=16)
        etf_bars["raw_sha256"] = etf_bars["source_id"]

    action_frames = []
    for symbol in ("SH510300", "SZ159915", "SH512890"):
        frame = fetch_actions(symbol, calendar, probes)
        if not frame.empty:
            action_frames.append(frame)
    etf_actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()

    # 股债比较与创业板估值: 明确探测失败并记录
    _record_probe(probes, "ratio_signals", "wind", "885007.WI", "not_available", 0,
                  note="Wind 混合债券型二级基金指数无免费合法访问权限, 未采购")
    _record_probe(probes, "index_valuation:399006:PB_MRQ", "cnindex", "getIndexDailyDataWithDataFormat",
                  "no_valuation_fields", 0,
                  note="国证官方日线接口仅返回价格量额, 无PB字段; 免费端未找到日频历史PB")
    _record_probe(probes, "index_valuation:000300:PE_TTM", "legulegu", "index-basic-pe", "not_probed", None,
                  note="本次构建未请求该接口；不可把过去观察写成本次实测")
    _record_probe(probes, "index_valuation:399006:PB_MRQ", "danjuan", "index_eva/pb_history", "not_probed", None,
                  note="本次构建未请求该接口；历史可用性须另存实际响应后判断")

    version = f"vb-{stamp}"
    snapshot_dir = data_root / "snapshots" / version
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files: List[Dict[str, object]] = []
    payloads = {
        "index_price": index_price,
        "index_valuation": index_valuation,
        "etf_bars": etf_bars,
        "etf_actions": etf_actions,
        "dividend_yield": dividend_yield,
    }
    for name, frame in payloads.items():
        path = snapshot_dir / f"{name}.csv"
        if frame is None or frame.empty:
            continue
        extract_path = raw_dir / f"{name}_provider_extract.csv"
        frame.drop(columns=["raw_sha256"], errors="ignore").to_csv(extract_path, index=False, encoding="utf-8")
        if "raw_sha256" in frame.columns:
            frame["raw_sha256"] = sha256_file(extract_path)
        frame.to_csv(path, index=False, encoding="utf-8")
        files.append({"name": name, "path": path.name, "sha256": sha256_file(path), "rows": int(len(frame))})
    empty_names = [name for name, frame in payloads.items() if frame is None or frame.empty]

    manifest = {
        "data_version": version,
        "dataset_kind": "real",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "calendar": {"exchange": "XSHG", "start": "2010-01-01", "end": requested_end},
        "files": files,
        "empty_files": empty_names,
        "sources": [
            {"series": "index_price:000300", "provider": "csindex_index_perf", "official": True,
             "availability_basis": "assumed_next_session_19", "evidence_grade": "historical_reconstructed",
             "pit_verified": False, "note": "指数提供方官方日线, 历史序列为今日下载, 缺当时发布快照"},
            {"series": "index_price:399006", "provider": "cnindex_index_daily", "official": True,
             "availability_basis": "assumed_next_session_19", "evidence_grade": "historical_reconstructed",
             "pit_verified": False},
            {"series": "etf_bars:SH510300", "provider": "project_formal_duckdb", "official": False,
             "availability_basis": "session_close_16h", "evidence_grade": "historical_reconstructed",
             "pit_verified": False, "note": "项目已审计原始不复权日线, 只读复用"},
            {"series": "etf_bars:SZ159915", "provider": "project_formal_duckdb", "official": False,
             "availability_basis": "session_close_16h", "evidence_grade": "historical_reconstructed",
             "pit_verified": False},
            {"series": "etf_bars:SH512890", "provider": "eastmoney_fund_etf_hist_em", "official": False,
             "availability_basis": "session_close_16h", "evidence_grade": "historical_reconstructed",
             "pit_verified": False},
            {"series": "etf_actions:*", "provider": "sina_hfq_js", "official": False,
             "availability_basis": "ex_date_only", "evidence_grade": "historical_reconstructed",
             "pit_verified": False, "note": "仅除息日与累计分红, 登记日/支付日为研究假设"},
            {"series": "index_valuation:000300:PE_CSINDEX_PE1/PE2", "provider": "csindex_indicator_xls",
             "official": True, "availability_basis": "assumed_next_session_19",
             "evidence_grade": "official_rolling_20_sessions", "pit_verified": False,
             "note": "官方估值文件仅20个交易日, 不构成日频历史"},
        ],
        "assumptions": [
            "全部历史观测的可得时间按 assumed_next_session_19 处理: 交易日 d 的观测假设在下一个交易日19:00可用。",
            "ETF 分红登记日按除息日前一交易日、支付日按除息后第5个交易日建模, 为研究假设而非公告事实。",
            "ETF 成交量由项目库的'手'按1手=100份换算; 金额单位为元。",
            "中证官方估值文件市盈率1/2 的财务口径未确认TTM, 因此单列为 PE_CSINDEX_* 而非 PE_TTM。",
        ],
        "probes": probes,
        "usage": "research_only",
    }
    manifest["raw_evidence_directory"] = str(raw_dir.resolve())
    manifest["raw_hash_basis"] = "provider_extract_csv; public HTTP responses archived separately when available"
    if not index_price.empty:
        for source in manifest["sources"]:
            if str(source.get("series", "")).startswith("index_price:"):
                code = source["series"].split(":")[1]
                providers = sorted(index_price.loc[index_price["index_id"].eq(code), "source_id"].unique())
                source["provider"] = ";".join(providers)
                source["official"] = bool(providers) and all(p in ("csindex_index_perf", "cnindex_index_daily") for p in providers)
                source["note"] = "实际下载源见provider及原始证据；历史重建，不宣称原始PIT快照"
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (raw_dir / "probes.json").write_text(
        json.dumps(probes, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest_path
