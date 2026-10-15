"""收盘信号→次日开盘的事件回放；使用共享组合与整数订单规则。"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from src.portfolio import target_weights, buy_notional_budget
from src.formal_orders import plan_orders


def simulate(market, scores, policy, *, mode="rank", slippage_multiplier=1.,
             capital_multiplier=1., fill_fraction=1., delay=1, observer=None, state=None, order_planner=None,
             unavailable_dates=(), holdings_log=None):
    if not 0 < fill_fraction <= 1 or delay < 1:
        raise ValueError("成交比例与执行延迟无效")
    close, adjclose = market.wide("close"), market.wide("adj_close")
    opened, factors, volume = market.wide("open"), market.wide("adj_factor"), market.wide("volume")
    close_mark, adj_mark = close.ffill(limit=5), adjclose.ffill(limit=5)
    returns = adjclose.pct_change(fill_method=None)
    eligible = market.eligibility.pivot(index="date", columns="symbol", values="eligible")
    adv = market.eligibility.pivot(index="date", columns="symbol", values="avg_amount")
    signal_dates = scores.index.get_level_values("datetime")
    by_day = {day: part.droplevel("datetime") for day, part in scores.groupby(level="datetime")}
    calendar = market.calendar
    start = max(int(calendar.searchsorted(signal_dates.min())), 1)
    cash = policy["initial_capital"] * capital_multiplier
    previous_nav = cash
    units, bought = {}, {}
    exiting = set()
    queue, records, trades = {}, [], []
    issues = {"suspended_or_missing": 0, "price_limit": 0, "capacity_clipped": 0, "cash_clipped": 0, "risk_clipped": 0}
    for i in range(start, len(calendar)):
        date = calendar[i]
        daily_cost = daily_turnover = 0.
        for order in queue.pop(i, []):
            symbol, side = order["symbol"], order["side"]
            price, factor = opened.at[date, symbol], factors.at[date, symbol]
            if not np.isfinite(price) or price <= 0 or not np.isfinite(factor) or volume.at[date, symbol] <= 0:
                issues["suspended_or_missing"] += 1
                continue
            threshold = .195 if symbol in ("SH588000", "SZ159915") else .095
            if abs(price / order["reference_price"] - 1) >= threshold:
                issues["price_limit"] += 1
                continue
            quantity = math.floor(order["quantity"] * fill_fraction / 100) * 100
            if side == 'sell' and order.get('full_exit') and fill_fraction == 1:
                quantity = units.get(symbol, 0) * factor
            if side == "sell" and order["target_weight"] == 0 and order["quantity"] < 100 and not order.get('full_exit'):
                quantity = order["quantity"]  # 清仓零股一次性卖出，避免复权单位产生永不退出的尾仓。
            capacity = math.floor(order["notional_limit"] / price / 100) * 100
            if quantity > capacity:
                issues["capacity_clipped"] += 1
                quantity = capacity
            if side == "sell":
                quantity = min(quantity, units.get(symbol, 0) * factor)
            if quantity <= 0:
                continue
            participation = quantity * price / max(order["adv"], 1.)
            slip = policy["slippage"] * slippage_multiplier + policy["impact_coefficient"] * math.sqrt(participation)
            fill_price = price * (1 + slip if side == "buy" else 1 - slip)
            if side == "buy":
                if policy.get("conservative_buy_budget"):
                    actual_prices = close_mark.iloc[i-1].to_dict()
                    actual_prices.update(opened.loc[date].dropna().to_dict())
                    actual_qty = {s: u * (factors.at[date, s] if np.isfinite(factors.at[date, s]) else factors.iloc[i-1][s]) for s, u in units.items()}
                    room = buy_notional_budget(symbol, actual_qty, cash, actual_prices, policy)
                    capped = math.floor(room / (fill_price * (1 + policy["commission"])) / 100) * 100
                    if quantity > capped:
                        issues["risk_clipped"] += 1
                        quantity = capped
                affordable = math.floor(max(0., cash - policy["minimum_commission"]) / (fill_price * (1 + policy["commission"])) / 100) * 100
                if quantity > affordable:
                    issues["cash_clipped"] += 1
                    quantity = affordable
            if quantity <= 0:
                continue
            amount = quantity * fill_price
            fee = max(policy["minimum_commission"], amount * policy["commission"])
            direction = 1 if side == "buy" else -1
            cash -= direction * amount + fee
            if side == "buy" and units.get(symbol, 0) <= 1e-9:
                bought[symbol] = i
            units[symbol] = units.get(symbol, 0) + direction * quantity / factor
            if units[symbol] < 1e-8:
                units.pop(symbol, None)
                bought.pop(symbol, None)
                exiting.discard(symbol)
            elif side == "sell" and order["target_weight"] <= 1e-9:
                exiting.add(symbol)
            daily_cost += fee + quantity * price * slip
            daily_turnover += quantity * price
            trades.append({"date": str(date.date()), "signal_date": order["signal_date"],
                           "symbol": symbol, "side": side, "quantity": quantity, "price": fill_price,
                           "commission": fee, "slippage_cost": quantity * price * slip,
                           "participation": participation})
        values = {}
        for symbol, unit in units.items():
            price = adj_mark.at[date, symbol]
            if not np.isfinite(price):
                raise ValueError(f"{date.date()} {symbol} 持仓连续缺价，禁止虚构估值")
            values[symbol] = unit * price
        nav = cash + sum(values.values())
        if cash < -1e-6 or nav <= 0:
            raise ValueError("模拟账户现金/净值异常")
        net_return = nav / previous_nav - 1
        cost = daily_cost / previous_nav
        records.append({"date": date, "account": nav, "return": net_return + cost,
                        "cost": cost, "turnover": daily_turnover / previous_nav,
                        "cash": cash, "value": sum(values.values())})
        if holdings_log is not None:
            for symbol, value in values.items():
                holdings_log.append({'date': date, 'symbol': symbol,
                                     'quantity': value / close_mark.at[date, symbol], 'value': value})
        current = {s: v / nav for s, v in values.items()}
        if date in by_day:
            candidates = eligible.columns[eligible.loc[date].fillna(False)].tolist()
            targets = {} if date in unavailable_dates else target_weights(by_day[date], candidates, current,
                         {s: i - bought[s] + 1 for s in units}, returns.loc[:date], policy, mode=mode, exiting=exiting)
            # 复权单位用于历史全收益估值，按当日原价折算可交易份额并取整。
            quantities = {s: values[s] / close_mark.at[date, s] for s in values}
            orders = (order_planner or plan_orders)(targets, quantities, cash, close_mark.loc[date], adv.loc[date], policy)
            for order in orders:
                order.update({"signal_date": str(date.date()), "adv": float(adv.at[date, order["symbol"]])})
            if observer is not None:
                observer(date, quantities.copy(), cash, bought.copy(), exiting.copy(), targets, orders)
            if i + delay < len(calendar):
                queue.setdefault(i + delay, []).extend(orders)
        previous_nav = nav
    report = pd.DataFrame(records).set_index("date")
    benchmark = adjclose["SH510300"].pct_change(fill_method=None).reindex(report.index).fillna(0.)
    report["bench"] = benchmark
    if state is not None:
        state.update(cash=cash, quantities={s: u * factors.at[calendar[-1], s] for s, u in units.items()},
                     bought=bought, exiting=sorted(exiting))
    return report, pd.DataFrame(trades), issues
