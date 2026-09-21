"""下载并冻结独立美元/人民币标普基准；不向策略提供境外当日收盘信号。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[2]


def parse_chart(data):
    dates = (pd.to_datetime(data["timestamp"], unit="s", utc=True)
             .tz_convert(data["meta"]["exchangeTimezoneName"]).normalize().tz_localize(None))
    quote = data["indicators"]["quote"][0]
    frame = pd.DataFrame({k: quote[k] for k in ("open", "high", "low", "close")}, index=dates)
    adjusted = data["indicators"].get("adjclose")
    frame["adjusted_close"] = adjusted[0]["adjclose"] if adjusted else frame["close"]
    frame.index.name = "date"
    frame = frame.dropna(subset=["close", "adjusted_close"]).sort_index()
    if frame.index.has_duplicates or (frame[["close", "adjusted_close"]] <= 0).any().any():
        raise ValueError("基准日期重复或存在非正价格")
    if not np.isfinite(frame[["close", "adjusted_close"]]).all().all():
        raise ValueError("基准价格非有限")
    return frame


def fetch_benchmarks(start="2015-01-01", end="2026-09-11"):
    directory = BASE / "output/sp500_exploration/benchmarks" / pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    records, frames = {}, {}
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
              "period2": int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp()),
              "interval": "1d", "events": "div,splits"}
    for key, symbol in (("spy", "SPY"), ("sp500_price", "%5EGSPC"), ("usdcny", "CNY=X")):
        with requests.Session() as session:
            response = session.get("https://query2.finance.yahoo.com/v8/finance/chart/" + symbol,
                                   params=params, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        raw = response.content
        (directory / f"{key}.json").write_bytes(raw)
        chart = response.json()["chart"]["result"][0]
        frame = parse_chart(chart).loc[start:end]
        if len(frame) < 1200 or frame.index[-1] < pd.Timestamp(end):
            raise ValueError(f"{key}基准覆盖不足")
        frame.to_csv(directory / f"{key}.csv")
        records[key] = {"url": response.url, "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "raw_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(frame),
                        "start": str(frame.index[0].date()), "end": str(frame.index[-1].date()),
                        "currency": chart["meta"]["currency"]}
        frames[key] = frame
        if key == "spy":
            dividends = pd.DataFrame(list(chart.get("events", {}).get("dividends", {}).values()))
            dividends["date"] = (pd.to_datetime(dividends["date"], unit="s", utc=True)
                                 .dt.tz_convert(chart["meta"]["exchangeTimezoneName"]).dt.tz_localize(None).dt.normalize())
            dividends.to_csv(directory / "spy_dividends.csv", index=False)
            dividend = dividends.set_index("date")["amount"].reindex(frame.index, fill_value=0)
            cash_return = (frame["close"] + dividend) / frame["close"].shift(1)
            cash_total = cash_return.iloc[1:].prod()
            adjusted_total = frame.adjusted_close.iloc[-1] / frame.adjusted_close.iloc[0]
            error = abs(cash_total / adjusted_total - 1)
            records[key]["cash_reinvestment_vs_adjusted_error"] = float(error)
            records[key]["dividend_events"] = len(dividends)
            if error > .005:
                raise ValueError("SPY分红递推与供应商复权收益不一致")
    # 独立供应商只读交叉核对近期标普价格，不能将其当作含分红收益。
    with requests.Session() as session:
        session.trust_env = False
        response = session.get("https://web.ifzq.gtimg.cn/appstock/app/usfqkline/get",
                               params={"param": f"us.INX,day,{start},{end},640,qfq"}, timeout=(8, 20))
        response.raise_for_status()
    (directory / "sp500_tencent.json").write_bytes(response.content)
    rows = response.json()["data"]["us.INX"]["day"]
    independent = pd.Series({pd.Timestamp(r[0]): float(r[2]) for r in rows})
    comparison = pd.concat([frames["sp500_price"].close, independent.rename("independent")], axis=1).dropna()
    error = (comparison.close / comparison.independent - 1).abs().max()
    if len(comparison) < 100 or error > .001:
        raise ValueError("标普价格交叉校验失败")
    manifest = {"sources": records, "price_crosscheck": {"rows": len(comparison), "max_relative_error": float(error)},
                "note": "SPY调整收盘为含分红、已包含基金内部费用的代理；USD/CNY为每美元人民币。境外日期收盘只用于事后比较，绝不进入当日中国市场信号。",
                "directory": str(directory)}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory.parent / "latest.json").write_text(json.dumps({"directory": str(directory)}, indent=2), encoding="utf-8")
    return directory, manifest


def aligned_wealth(directory, calendar):
    """按报告日期作向后匹配；最多7个自然日报价空档，不跨数据末尾填充。"""
    directory = Path(directory)
    frames = {k: pd.read_csv(directory / f"{k}.csv", index_col="date", parse_dates=True)
              for k in ("spy", "sp500_price", "usdcny")}
    series = {}
    for key, frame in frames.items():
        column = "adjusted_close" if key == "spy" else "close"
        if calendar[-1] > frame.index[-1] or calendar[0] < frame.index[0]:
            raise ValueError(f"{key}与策略区间不一致")
        # tolerance按自然日限定跨周末/假期填充，不能静默延长行情。
        series[key] = frame[column].reindex(calendar, method="ffill", tolerance=pd.Timedelta(days=7))
        if series[key].isna().any():
            raise ValueError(f"{key}基准存在过长缺口")
    wealth = pd.DataFrame({"sp500_price_usd": series["sp500_price"], "spy_total_usd": series["spy"],
                           "spy_total_cny": series["spy"] * series["usdcny"]})
    return wealth / wealth.iloc[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default="2026-09-11")
    args = parser.parse_args()
    directory, manifest = fetch_benchmarks(end=args.end)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
