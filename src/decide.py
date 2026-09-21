"""正式收盘信号→下一交易日开盘的模拟/人工订单建议，不自动下单或覆盖持仓。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from filelock import FileLock
import config
from backend.artifacts import write_json_atomic
from src.account import read_account
from src.market import load_market
from src.models import predict_bundle
from src.portfolio import target_weights, plan_orders
from src.runs import load_production, code_fingerprint, read_protocol
from src.utils import load_names, symbol_to_code


def next_session(day):
    import exchange_calendars as xcals
    day = pd.Timestamp(day).normalize()
    calendar = xcals.get_calendar("XSHG")
    if not calendar.is_session(day):
        raise ValueError(f"{day.date()} 不是交易所交易日")
    return calendar.next_session(day).strftime("%Y-%m-%d")


def validate_readiness(bundle, manifest, signal_date, *, now=None, paper=False):
    now = now if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Shanghai")
    now = now.tz_convert("Asia/Shanghai")
    signal = pd.Timestamp(signal_date).normalize()
    if signal > now.tz_localize(None).normalize():
        raise ValueError("信号日期在未来")
    if signal == now.tz_localize(None).normalize() and now.strftime("%H:%M") < config.DECISION_RUN_TIME:
        raise ValueError(f"今日正式日线尚未完成，须在 {config.DECISION_RUN_TIME} 后生成信号")
    if pd.Timestamp(bundle["train_label_end"]) > signal or pd.Timestamp(bundle["trained_as_of"]) > signal:
        raise ValueError("模型训练使用了信号日之后的标签，禁止历史穿越")
    if (signal - pd.Timestamp(bundle["trained_as_of"])).days > config.MODEL_STALE_DAYS:
        raise ValueError("生产模型训练数据已过期，请重新训练")
    if not paper:
        if (now.tz_localize(None).normalize() - signal).days > config.MAX_DATA_STALE_DAYS:
            raise ValueError("行情已过期，停止生成可执行建议")
        if not manifest.get("investment_gate", {}).get("approved"):
            raise ValueError("投资准入未通过，只能显式使用 --paper 进行模拟研究")
        execution = pd.Timestamp(next_session(signal), tz="Asia/Shanghai") + pd.Timedelta(hours=9, minutes=25)
        if now >= execution:
            raise ValueError("计划开盘时点已过，不能执行过期信号")
    return {"paper": paper, "data_age_calendar_days": (now.tz_localize(None).normalize() - signal).days,
            "signal_date": str(signal.date()), "execution_date": next_session(signal)}


def generate_decision(account_path=None, *, as_of=None, paper=False, output_dir=None):
    account_path = Path(account_path or config.POSITIONS_FILE)
    with FileLock(str(account_path) + ".lock"):
        account = read_account(account_path)
    bundle, manifest = load_production()
    if bundle["code_fingerprint"] != code_fingerprint()["fingerprint"]:
        raise ValueError("投研代码已变化，须重新训练并验收后生成建议")
    protocol = manifest["protocol"]
    if protocol != read_protocol():
        raise ValueError("研究协议已变化，须重新训练并验收后生成建议")
    market = load_market(as_of=as_of, horizon=protocol["label_horizon"])
    date = market.calendar[-1]
    readiness = validate_readiness(bundle, manifest, date, paper=paper)
    if account["mode"] != "paper":
        raise ValueError("当前仅接入模拟/人工核对账本，无真实券商执行通道")
    latest = market.features.xs(date, level="datetime")
    score = predict_bundle(bundle, latest)
    eligible = market.eligibility[(market.eligibility["date"] == date) & market.eligibility["eligible"]]
    prices = market.wide("close").loc[date].dropna().to_dict()
    adv = market.eligibility[market.eligibility["date"] == date].set_index("symbol")["avg_amount"].to_dict()
    quantities = {s: h["quantity"] for s, h in account["holdings"].items()}
    for symbol in quantities:
        if symbol not in prices:
            raise ValueError(f"持仓 {symbol} 缺少当日价格，停止估值")
    nav = account["cash"] + sum(quantities[s] * prices[s] for s in quantities)
    current = {s: quantities[s] * prices[s] / nav for s in quantities}
    held_days = {}
    for s, h in account["holdings"].items():
        bought = pd.Timestamp(h["buy_date"])
        if bought > date:
            raise ValueError("账户持仓买入日在信号之后，不能用于历史回放")
        held_days[s] = int(((market.calendar >= bought) & (market.calendar <= date)).sum())
    history = market.wide("adj_close").pct_change(fill_method=None).loc[:date]
    targets = target_weights(score, eligible["symbol"], current, held_days, history, protocol,
                             exiting=[s for s, h in account["holdings"].items() if h.get("exit_pending")])
    orders = plan_orders(targets, quantities, account["cash"], prices, adv, protocol)
    identity = json.dumps({"run_id": bundle["run_id"], "date": str(date.date()), "revision": account["revision"],
                           "account": str(account_path.resolve()), "targets": targets, "data": market.fingerprint}, sort_keys=True)
    decision_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    for i, order in enumerate(orders):
        order["order_id"] = f"{decision_id}-{i+1}"
    payload = {"schema_version": 2, "decision_id": decision_id, "run_id": bundle["run_id"],
               "account_revision": account["revision"], "signal_date": str(date.date()),
               "execution_date": readiness["execution_date"], "execution_window": "next_session_open",
               "mode": "paper", "orders": orders, "target_weights": targets, "account_nav": nav,
               "data_fingerprint": market.fingerprint, "investment_gate": manifest["investment_gate"],
               "readiness": readiness, "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()}
    directory = Path(output_dir or config.OUTPUT_DIR / "decisions")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{decision_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if output_dir is None:
            from src.forward import freeze_signal
            freeze_signal(existing, manifest, account_path)
            write_json_atomic(config.OUTPUT_DIR / "latest_decision.json", existing)
        return existing, path
    write_json_atomic(path, payload)
    names = load_names()
    rows = [{"操作": "买入" if o["side"] == "buy" else "卖出", "代码": o["symbol"],
             "名称": names.get(symbol_to_code(o["symbol"]), ""), "份额": o["quantity"],
             "参考收盘价": o["reference_price"], "目标权重": o["target_weight"],
             "订单编号": o["order_id"]} for o in orders]
    pd.DataFrame(rows, columns=["操作", "代码", "名称", "份额", "参考收盘价", "目标权重", "订单编号"]).to_csv(
        directory / f"{decision_id}.csv", index=False, encoding="utf-8-sig")
    if output_dir is None:
        from src.forward import freeze_signal
        freeze_signal(payload, manifest, account_path)
        write_json_atomic(config.OUTPUT_DIR / "latest_decision.json", payload)
    return payload, path


def main():
    parser = argparse.ArgumentParser(description="收盘后生成下一交易日开盘订单建议；不会下单")
    parser.add_argument("--paper", action="store_true", help="仅模拟研究，可查看未通过准入的策略建议")
    parser.add_argument("--account", type=Path, default=config.POSITIONS_FILE)
    parser.add_argument("--as-of", help="模拟历史信号日期，必须不早于模型训练信息截止日")
    parser.add_argument("--no-refresh", action="store_true", help="兼容参数：本命令只读正式行情，不自动抓取")
    parser.add_argument("--apply", action="store_true", help="已停用；请使用src.account apply-fills逐笔对账")
    args = parser.parse_args()
    if args.apply:
        raise SystemExit("--apply 已停用：不能把建议当作成交。请使用 python -m src.account apply-fills --decision ... --fills ...")
    if args.as_of and not args.paper:
        raise SystemExit("历史回放必须显式 --paper")
    try:
        decision, path = generate_decision(args.account, as_of=args.as_of, paper=args.paper)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"模拟决策 {decision['signal_date']} → {decision['execution_date']} 开盘；{len(decision['orders'])}个订单")
    print(f"建议已保存 {path}；持仓未变化，只能根据真实/模拟成交回报记账")


if __name__ == "__main__":
    main()
