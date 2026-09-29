"""前瞻模拟观察：执行前冻结信号，执行日收盘后记账，禁止历史回填冒充前瞻。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from filelock import FileLock
import config
from src.artifacts import write_json_atomic
from src.account import read_account
from src.market import load_market
from src.runs import load_production


def _now():
    return pd.Timestamp.now(tz="Asia/Shanghai")


def observation_dir(protocol_hash, code_hash, account_path=None):
    account = str(Path(account_path or config.POSITIONS_FILE).resolve())
    account_key = hashlib.sha256(account.encode()).hexdigest()[:16]
    return config.OUTPUT_DIR / "forward" / f"{protocol_hash}-{code_hash}" / account_key


def freeze_signal(decision, manifest, account_path=None):
    now = _now()
    execution = pd.Timestamp(decision["execution_date"], tz="Asia/Shanghai") + pd.Timedelta(hours=9, minutes=25)
    if now >= execution or decision["readiness"]["data_age_calendar_days"] > 3:
        return False
    directory = observation_dir(manifest["protocol_hash"], manifest["code"]["fingerprint"], account_path)
    path = directory / "signals" / f"{decision['execution_date']}.json"
    payload = {"frozen_at": now.isoformat(), "decision": decision}
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["decision"]["decision_id"] != decision["decision_id"]:
                raise ValueError("该执行日已有冻结信号，禁止事后替换")
        else:
            write_json_atomic(path, payload)
    return True


def record_close(account_path=None):
    now = _now()
    if now.strftime("%H:%M") < config.DECISION_RUN_TIME:
        raise ValueError("必须等正式收盘数据完成后记录观察")
    _, manifest = load_production()
    directory = observation_dir(manifest["protocol_hash"], manifest["code"]["fingerprint"], account_path)
    day = str(now.date())
    signal_path = directory / "signals" / f"{day}.json"
    if not signal_path.exists():
        raise ValueError("执行前未冻结当日信号，不能补记为前瞻观察")
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    if pd.Timestamp(signal["frozen_at"]) >= now.normalize() + pd.Timedelta(hours=9, minutes=25):
        raise ValueError("信号冻结晚于计划开盘时间")
    market = load_market(as_of=day, horizon=manifest["protocol"]["label_horizon"])
    if str(market.calendar[-1].date()) != day:
        raise ValueError("当日正式行情尚未齐备，不能估算观察净值")
    account = read_account(account_path)
    prices = market.wide("close").iloc[-1].dropna().to_dict()
    nav = account["cash"]
    for symbol, holding in account["holdings"].items():
        if symbol not in prices:
            raise ValueError(f"{symbol} 当日缺价，停止记录")
        nav += holding["quantity"] * prices[symbol]
    trades = [v["fill"] for v in account["executions"].values()
              if v["fill"]["decision_id"] == signal["decision"]["decision_id"]]
    outside_open = sum(pd.Timestamp(t["executed_at"]).tz_convert("Asia/Shanghai").strftime("%H:%M") > "09:35" for t in trades)
    completed = {o["order_id"]: sum(t["quantity"] for t in trades if t["order_id"] == o["order_id"])
                 / o["quantity"] for o in signal["decision"]["orders"]}
    min_fill_ratio = min(completed.values(), default=1.)
    unrelated = any(str(pd.Timestamp(v["fill"]["executed_at"]).tz_convert("Asia/Shanghai").date()) == day
                    and v["fill"]["decision_id"] != signal["decision"]["decision_id"]
                    for v in account["executions"].values())
    receipt = {"date": day, "recorded_at": now.isoformat(), "nav": nav,
               "opening_reference_nav": signal["decision"]["account_nav"],
               "decision_id": signal["decision"]["decision_id"], "account_revision": account["revision"],
               "account": account, "executions_outside_open": outside_open,
               "execution_aligned": outside_open == 0 and not unrelated
                    and min_fill_ratio >= manifest["protocol"]["min_forward_fill_ratio"],
               "minimum_order_fill_ratio": min_fill_ratio,
               "data_fingerprint": market.fingerprint}
    destination = directory / "closes" / f"{day}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            old = json.loads(destination.read_text(encoding="utf-8"))
            if old["account_revision"] != receipt["account_revision"] or old["nav"] != nav:
                raise ValueError("观察日已有记录且账户发生变化，须人工审计，禁止覆盖")
            return destination
        write_json_atomic(destination, receipt)
    return destination


def forward_evidence(protocol_hash, code_hash, account_path=None):
    directory = observation_dir(protocol_hash, code_hash, account_path)
    rows = []
    for path in sorted((directory / "closes").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        signal_path = directory / "signals" / path.name
        if not signal_path.exists():
            continue
        signal = json.loads(signal_path.read_text(encoding="utf-8"))
        opening = pd.Timestamp(receipt["date"], tz="Asia/Shanghai") + pd.Timedelta(hours=9, minutes=25)
        if pd.Timestamp(signal["frozen_at"]) >= opening or receipt["decision_id"] != signal["decision"]["decision_id"]:
            continue
        recorded = pd.Timestamp(receipt["recorded_at"]).tz_convert("Asia/Shanghai")
        if (str(recorded.date()) != receipt["date"] or recorded.strftime("%H:%M") < config.DECISION_RUN_TIME
                or recorded > _now()
                or signal["decision"]["execution_date"] != receipt["date"]):
            continue
        rows.append(receipt)
    if not rows:
        return {"days": 0, "return": None, "max_drawdown": None, "execution_aligned": False}
    import exchange_calendars as xcals
    expected = xcals.get_calendar("XSHG").sessions_in_range(rows[0]["date"], rows[-1]["date"])
    continuous = [d.strftime("%Y-%m-%d") for d in expected] == [r["date"] for r in rows]
    # 漏日或账户中途重置不能拼接成连续前瞻净值。
    reconciled = all(abs(rows[i]["opening_reference_nav"] - rows[i-1]["nav"]) <= .01
                     for i in range(1, len(rows)))
    wealth = pd.Series([r["nav"] for r in rows], dtype=float) / rows[0]["opening_reference_nav"]
    return {"days": len(rows), "return": float(wealth.iloc[-1]-1),
            "max_drawdown": float((wealth/wealth.cummax().clip(lower=1)-1).min()),
            "continuous": continuous, "account_reconciled": reconciled,
            "execution_aligned": continuous and reconciled and all(r.get("execution_aligned", False) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description="记录当天真实前瞻模拟观察，不能历史回填")
    parser.add_argument("--account", type=Path, default=config.POSITIONS_FILE)
    args = parser.parse_args()
    print(record_close(args.account))


if __name__ == "__main__":
    main()
