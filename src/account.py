"""模拟/人工账户：显式初始化、严格状态检查、幂等成交回报记账。无券商下单。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from filelock import FileLock
import config
from src.artifacts import write_json_atomic


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为数值")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name}无效")
    return number


def validate_account(state):
    if not isinstance(state, dict) or state.get("schema_version") != 2:
        raise ValueError("账户格式不完整；旧版仅代码列表不能代表真实持仓，请按份额/现金迁移")
    state["cash"] = _positive(state.get("cash"), "现金", allow_zero=True)
    if state.get("mode") != "paper":
        raise ValueError("仅支持模拟/人工核对账户")
    if not isinstance(state.get("holdings"), dict) or not isinstance(state.get("executions"), dict):
        raise ValueError("账户缺少持仓或成交明细")
    if isinstance(state.get("revision"), bool) or not isinstance(state.get("revision"), int) or state["revision"] < 0:
        raise ValueError("账户版本无效")
    for key in ("order_filled", "decision_revisions"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"账户缺少{key}明细")
    for symbol, holding in state["holdings"].items():
        if not isinstance(symbol, str) or not symbol.startswith(("SH", "SZ")) or len(symbol) != 8:
            raise ValueError("持仓代码无效")
        holding["quantity"] = _positive(holding["quantity"], "持仓份额")
        holding["average_cost"] = _positive(holding["average_cost"], "持仓成本")
        if pd.isna(pd.Timestamp(holding["buy_date"])):
            raise ValueError("买入日期无效")
    return state


def read_account(path=None):
    path = Path(path or config.POSITIONS_FILE)
    if not path.exists():
        raise ValueError("账户未初始化：请先核对现金与份额，或显式初始化模拟账户")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return validate_account(state)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"账户状态异常，停止决策：{exc}") from exc


def initialize(path, cash, holdings=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 2, "mode": "paper", "revision": 0, "cash": _positive(cash, "初始资金"),
             "holdings": holdings or {}, "executions": {}, "order_filled": {}, "decision_revisions": {},
             "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()}
    validate_account(state)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            raise ValueError("账户已存在，禁止初始化覆盖")
        write_json_atomic(path, state)
    return state


def apply_fills(path, decision, fills):
    """一个批次全部成功或全部不写；重复execution_id必须内容完全相同。"""
    path = Path(path)
    if not isinstance(fills, list) or not fills:
        raise ValueError("成交回报必须为非空JSON列表")
    with FileLock(str(path) + ".lock"):
        state = read_account(path)
        decision_id = decision.get("decision_id")
        if not decision_id or not decision.get("execution_date") or "account_revision" not in decision:
            raise ValueError("决策缺少版本或执行日期")
        orders = {order["order_id"]: order for order in decision["orders"]}
        known_decisions = state.setdefault("decision_revisions", {})
        is_new = decision_id not in known_decisions
        new_count = 0
        for fill in fills:
            execution_id = fill.get("execution_id")
            if not isinstance(execution_id, str) or not execution_id.strip():
                raise ValueError("成交缺少execution_id")
            fingerprint = hashlib.sha256(json.dumps(fill, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if execution_id in state["executions"]:
                if state["executions"][execution_id]["fingerprint"] != fingerprint:
                    raise ValueError("同一成交ID内容冲突")
                continue
            if is_new and decision["account_revision"] != state["revision"]:
                raise ValueError("账户已变化，旧决策不可执行，请重新生成建议")
            if not is_new and state.get("active_decision_id") != decision_id:
                raise ValueError("已有更新决策，拒绝继续执行过期决策")
            order = orders.get(fill.get("order_id"))
            if order is None or fill.get("decision_id") != decision_id:
                raise ValueError("成交未匹配订单/决策")
            symbol, side = order["symbol"], order["side"]
            if fill.get("symbol") != symbol or fill.get("side") != side:
                raise ValueError("成交方向或标的与订单不一致")
            executed = pd.Timestamp(fill["executed_at"])
            if executed.tzinfo is None:
                raise ValueError("成交时间必须携带时区")
            local = executed.tz_convert("Asia/Shanghai")
            if str(local.date()) != decision["execution_date"] or local > pd.Timestamp.now(tz="Asia/Shanghai"):
                raise ValueError("成交不在计划执行日或时间在未来")
            if local.strftime("%H:%M") < "09:25" or local.strftime("%H:%M") > "15:00":
                raise ValueError("成交时间不在交易时段")
            quantity = _positive(fill["quantity"], "成交份额")
            price = _positive(fill["price"], "成交价格")
            fee = _positive(fill["fee"], "成交费用", allow_zero=True)
            cumulative = state["order_filled"].get(order["order_id"], 0) + quantity
            if cumulative > order["quantity"] + 1e-8:
                raise ValueError("累计成交超过订单份额")
            holding = state["holdings"].get(symbol)
            if side == "buy":
                if quantity != int(quantity):
                    raise ValueError("买入份额必须为整数")
                spend = quantity * price + fee
                if spend > state["cash"] + 1e-8:
                    raise ValueError("成交后现金为负，拒绝记账")
                old_qty = holding["quantity"] if holding else 0
                old_cost = old_qty * holding["average_cost"] if holding else 0
                locked = 0 if not holding else (
                    holding.get("unsettled_quantity", old_qty) if
                    holding.get("last_buy_date", holding["buy_date"]) == decision["execution_date"] else 0)
                state["holdings"][symbol] = {"quantity": old_qty + quantity,
                    "average_cost": (old_cost + spend) / (old_qty + quantity),
                    "buy_date": holding["buy_date"] if holding else decision["execution_date"],
                    "last_buy_date": decision["execution_date"], "unsettled_quantity": locked + quantity}
                state["cash"] -= spend
            elif side == "sell":
                if holding is None or quantity > holding["quantity"] + 1e-8:
                    raise ValueError("卖出超过持仓份额")
                locked = (holding.get("unsettled_quantity", holding["quantity"]) if
                          holding.get("last_buy_date", holding["buy_date"]) == decision["execution_date"] else 0)
                if quantity > holding["quantity"] - locked + 1e-8:
                    raise ValueError("保守T+1约束：当日买入持仓不可当日卖出")
                state["cash"] += quantity * price - fee
                holding["quantity"] -= quantity
                if holding["quantity"] <= 1e-8:
                    del state["holdings"][symbol]
                elif order.get("target_weight") == 0:
                    holding["exit_pending"] = True
            else:
                raise ValueError("未知买卖方向")
            state["order_filled"][order["order_id"]] = cumulative
            state["executions"][execution_id] = {"fingerprint": fingerprint, "fill": fill}
            new_count += 1
        if new_count:
            known_decisions[decision_id] = decision["account_revision"]
            state["active_decision_id"] = decision_id
            state["revision"] += 1
            state["updated_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
            validate_account(state)
            write_json_atomic(path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description="模拟账户与人工成交对账；不会下单")
    parser.add_argument("--account", type=Path, default=config.POSITIONS_FILE)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--cash", type=float, required=True)
    init.add_argument("--holdings", type=Path, help="可选，核对后的份额/成本/买入日期JSON")
    apply = sub.add_parser("apply-fills")
    apply.add_argument("--decision", type=Path, required=True)
    apply.add_argument("--fills", type=Path, required=True)
    actions = sub.add_parser("apply-actions")
    actions.add_argument("--actions", type=Path, required=True, help="核对后的分红/份额折算/清盘回款事件JSON")
    args = parser.parse_args()
    if args.command == "init":
        holdings = json.loads(args.holdings.read_text(encoding="utf-8")) if args.holdings else None
        state = initialize(args.account, args.cash, holdings)
    elif args.command == "apply-fills":
        state = apply_fills(args.account, json.loads(args.decision.read_text(encoding="utf-8")), json.loads(args.fills.read_text(encoding="utf-8")))
    else:
        state = apply_actions(args.account, json.loads(args.actions.read_text(encoding="utf-8")))
    print(f"模拟账户revision={state['revision']}，现金={state['cash']:.2f}，持仓={len(state['holdings'])}只")


def apply_actions(path, actions):
    """根据已核对回单记分红、拆分或清盘回款；不从复权因子猜测现金流。"""
    if not isinstance(actions, list) or not actions:
        raise ValueError("公司行动必须为非空列表")
    path = Path(path)
    with FileLock(str(path) + ".lock"):
        state = read_account(path)
        events = state.setdefault("corporate_actions", {})
        changed = False
        for event in actions:
            event_id, source = event.get("event_id"), event.get("source")
            if not isinstance(event_id, str) or not event_id or not isinstance(source, str) or not source:
                raise ValueError("公司行动必须包含事件ID及核对来源")
            signature = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
            if event_id in events:
                if events[event_id]["fingerprint"] != signature:
                    raise ValueError("公司行动ID内容冲突")
                continue
            date = pd.Timestamp(event["effective_date"])
            if pd.isna(date) or date > pd.Timestamp.now().normalize():
                raise ValueError("公司行动日期无效或在未来")
            symbol, kind = event.get("symbol"), event.get("type")
            if kind == "cash_dividend":
                state["cash"] += _positive(event["cash_amount"], "实际到账分红")
            elif kind == "split":
                if symbol not in state["holdings"]:
                    raise ValueError("份额折算标的不在持仓中")
                ratio = _positive(event["ratio"], "份额折算比例")
                state["holdings"][symbol]["quantity"] *= ratio
                state["holdings"][symbol]["average_cost"] /= ratio
                if "unsettled_quantity" in state["holdings"][symbol]:
                    state["holdings"][symbol]["unsettled_quantity"] *= ratio
            elif kind == "delist_settlement":
                if symbol not in state["holdings"]:
                    raise ValueError("清盘标的不在持仓中")
                state["cash"] += _positive(event["cash_amount"], "清盘实际回款", allow_zero=True)
                del state["holdings"][symbol]
            else:
                raise ValueError("仅支持cash_dividend/split/delist_settlement")
            events[event_id] = {"fingerprint": signature, "event": event}
            changed = True
        if changed:
            state["revision"] += 1
            state["active_decision_id"] = None
            state["updated_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
            validate_account(state)
            write_json_atomic(path, state)
        return state


if __name__ == "__main__":
    main()
