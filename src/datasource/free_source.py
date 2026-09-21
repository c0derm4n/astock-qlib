"""免费数据：腾讯不复权日线 + 新浪公司行动，拒绝拼接加法后复权序列。"""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
import requests


def _get(url, params=None):
    with requests.Session() as session:
        # 本机环境代理无法连接这些公开行情端点；直接访问，不传任何授权信息。
        session.trust_env = False
        response = session.get(url, params=params, timeout=(8, 20))
        response.raise_for_status()
        return response.text


def fetch_recent_bars(symbol, end):
    """腾讯最多640根，仅用于有历史基线的增量更新；量为手、额由万元转元。"""
    code = symbol.lower()
    text = _get("https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
                {"param": f"{code},day,,{pd.Timestamp(end).strftime('%Y-%m-%d')},640,"})
    payload = json.loads(text)
    rows = payload.get("data", {}).get(code, {}).get("day")
    if payload.get("code") != 0 or not rows:
        raise ValueError(f"{symbol} 免费原始日线缺失；不接受qfqday/hfqday回退")
    frame = pd.DataFrame([{ "symbol": symbol, "date": r[0], "open": float(r[1]),
        "close": float(r[2]), "high": float(r[3]), "low": float(r[4]),
        "volume": float(r[5]), "amount": float(r[8]) * 10000.} for r in rows])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    frame = frame[frame["date"] <= pd.Timestamp(end)].copy()
    frame["pct_chg"] = frame["close"].pct_change() * 100
    if (frame["volume"] < 0).any() or (frame["amount"] < 0).any():
        raise ValueError("免费行情量额为负")
    traded = frame["volume"] > 0
    vwap = frame.loc[traded, "amount"] / (frame.loc[traded, "volume"] * 100)
    # 日额有万元舍入；1%容差用于单位与结构校验，不替代交易价格。
    if ((vwap < frame.loc[traded, "low"] * .99) | (vwap > frame.loc[traded, "high"] * 1.01)).any():
        raise ValueError("免费行情成交量/金额单位校验失败")
    return frame.reset_index(drop=True)


def fetch_actions(symbol):
    text = _get(f"https://finance.sina.com.cn/realstock/company/{symbol.lower()}/hfq.js")
    # 只解析JSON，绝不eval远端脚本。
    prefix = f"var {symbol.lower()}hfq="
    if not text.strip().startswith(prefix):
        raise ValueError(f"{symbol} 公司行动返回格式未知")
    encoded = text.strip()[len(prefix):]
    data, end = json.JSONDecoder().raw_decode(encoded)
    if not re.fullmatch(r"\s*;?\s*(?:/\*[\s\S]*\*/)?\s*", encoded[end:]):
        raise ValueError("公司行动JSON之后出现未知脚本，拒绝解析")
    if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
        raise ValueError(f"{symbol} 公司行动资料为空，不能推断没有分红")
    frame = pd.DataFrame(data["data"])
    if not {"d", "s", "u"}.issubset(frame):
        raise ValueError("公司行动缺少日期、份额因子或累计分红")
    frame = frame.rename(columns={"d": "date", "s": "share_factor", "u": "cumulative_cash"})
    frame["date"] = pd.to_datetime(frame["date"])
    for field in ("share_factor", "cumulative_cash"):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    return frame.sort_values("date").reset_index(drop=True)


def aligned_increment(raw, actions, old):
    """以旧因子作锚，用核对到的分红事件递推；先验证历史重叠，再追加新日期。"""
    old = old.copy()
    old["date"] = pd.to_datetime(old["date"])
    merged = raw.merge(old, on=["symbol", "date"], suffixes=("_new", "_old"))
    if len(merged) < 20:
        raise ValueError("免费增量至少需要20根重叠日线，不能用于未审计的新标的全历史")
    overlap = merged.tail(60)
    price_error = float((overlap["close_new"] / overlap["close_old"] - 1).abs().max())
    if price_error > .005:
        raise ValueError(f"免费原价与旧数据不一致：{price_error:.3%}")
    first_date = overlap["date"].iloc[0]
    anchor = float(overlap["adj_factor"].iloc[0])
    rows = raw[raw["date"] >= first_date].copy()
    baseline_events = actions[actions["date"] <= first_date]
    if baseline_events.empty:
        raise ValueError("公司行动缺少锚点之前的累计状态")
    previous = baseline_events.iloc[-1]
    factor = anchor
    factors = []
    previous_close = None
    for row in rows.itertuples():
        events = actions[(actions["date"] > previous["date"]) & (actions["date"] <= row.date)]
        if len(events) > 1:
            raise ValueError("跨越多个公司行动且缺少逐日价格，停止拼接")
        for event in events.itertuples():
            split = event.share_factor / previous["share_factor"]
            dividend = event.cumulative_cash - previous["cumulative_cash"]
            if (not np.isfinite(split) or split <= 0 or dividend < -1e-8
                    or (abs(dividend) > 1e-8 and abs(split - 1) > 1e-8)):
                raise ValueError("复杂公司行动口径不支持自动拼接")
            factor *= split
            if dividend > 1e-8:
                if previous_close is None or previous_close <= dividend:
                    raise ValueError("分红缺少前收盘价或金额异常")
                factor *= previous_close / (previous_close - dividend)
            previous = pd.Series(event._asdict())
        factors.append(factor)
        previous_close = row.close
    rows["adj_factor"] = factors
    checked = rows[["date", "adj_factor"]].merge(overlap[["date", "adj_factor"]], on="date", suffixes=("_new", "_old"))
    factor_error = float((checked["adj_factor_new"] / checked["adj_factor_old"] - 1).abs().max())
    if factor_error > .003:
        raise ValueError(f"免费公司行动与旧因子不一致：{factor_error:.3%}")
    newer = rows[rows["date"] > old["date"].max()].copy()
    # 已有供应商因子有小数舍入，必须在最后重叠日精确衔接，避免边界虚假收益。
    last = checked.iloc[-1]
    newer["adj_factor"] *= last["adj_factor_old"] / last["adj_factor_new"]
    if not newer.empty:
        import exchange_calendars as xcals
        expected = xcals.get_calendar("XSHG").sessions_in_range(old["date"].max() + pd.Timedelta(days=1), newer["date"].max())
        if set(expected) != set(newer["date"]):
            raise ValueError("免费增量交易日缺失，需确认停牌等原始证据后再补齐")
    if not np.isfinite(rows[["open", "high", "low", "close", "adj_factor"]]).all().all():
        raise ValueError("免费数据有非有限价格或因子")
    return newer, {"overlap_days": len(overlap), "price_error": price_error, "factor_error": factor_error,
                   "new_rows": len(newer), "latest": str(rows["date"].max().date())}
