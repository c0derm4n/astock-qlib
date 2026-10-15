"""正式整数委托：抑制复权等值份额的无触发碎单。"""
import math
from src.portfolio import plan_orders as legacy_plan_orders


def plan_orders(targets, quantities, cash, prices, adv, policy):
    orders = legacy_plan_orders(targets, quantities, cash, prices, adv, policy)
    nav = cash + sum(q * prices[s] for s, q in quantities.items())
    result = []
    for order in orders:
        symbol = order['symbol']
        current = quantities.get(symbol, 0) * prices[symbol] / nav
        target = targets.get(symbol, 0)
        if abs(current - target) < 1e-10:
            continue  # 目标不变时，不能因 floor(nav*w/price) 的误差生成碎单。
        q = order['quantity']
        if order['side'] == 'sell' and target == 0 and q >= quantities.get(symbol, 0) - 1e-8:
            q = math.floor(q + 1e-8)
            # 复权代理账户在完全退出时注销小数等值份额；不是新增真实可交易份额。
            result.append({**order, 'quantity': float(q), 'full_exit': True})
            continue
        else:
            q = math.floor(q / 100) * 100
        if q > 0:
            result.append({**order, 'quantity': float(q)})
    return result
