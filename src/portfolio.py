"""回测与决策共用的组合规则和订单约束。所有输入必须在信号时点可知。"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import universe
from src.utils import symbol_to_code


def risk_group(symbol):
    code = symbol_to_code(symbol)
    groups = {
        "growth": {"159915", "588000", "159995", "512720", "515070", "512980", "515880", "515790", "515030"},
        "finance": {"512880", "512800"}, "health": {"512010", "512170"},
        "consumer": {"159928", "512690", "159996"},
        "cyclical": {"512400", "515220", "512200", "512580", "512660"},
    }
    for name, codes in groups.items():
        if code in codes:
            return name
    asset = universe.get_asset_class(code)
    return asset if asset != "equity" else "broad"


def constrain_weights(weights, returns, policy):
    weights = pd.Series(weights, dtype=float).clip(lower=0, upper=policy["max_weight"])
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0)
    for group in sorted({risk_group(s) for s in weights.index}):
        syms = [s for s in weights.index if risk_group(s) == group]
        total = weights.loc[syms].sum()
        if total > policy["max_group_weight"]:
            weights.loc[syms] *= policy["max_group_weight"] / total
    equity = [s for s in weights.index if universe.get_asset_class(symbol_to_code(s)) == "equity"]
    eq = weights.reindex(equity).sum()
    if eq > policy["max_equity_weight"]:
        weights.loc[equity] *= policy["max_equity_weight"] / eq
    if weights.sum() > policy["max_gross_weight"]:
        weights *= policy["max_gross_weight"] / weights.sum()
    if len(weights):
        history = returns.reindex(columns=weights.index).tail(policy["covariance_days"])
        covariance = history.cov(min_periods=20).fillna(0).to_numpy()
        for i, s in enumerate(weights.index):
            if history[s].count() < 20:
                covariance[i, i] = .4 ** 2 / policy["annualization_days"]
        variance = max(0., float(weights.to_numpy() @ covariance @ weights.to_numpy()))
        volatility = math.sqrt(variance * policy["annualization_days"])
        if volatility > policy["vol_target"]:
            weights *= policy["vol_target"] / volatility
    return weights[weights > 1e-9]


def target_weights(scores, eligible, current, holding_days, returns, policy, *, mode="rank", exiting=()):
    """最小持有期仅约束排名换仓，资格退出与风险缩减可以优先执行。"""
    current = pd.Series(current, dtype=float)
    allowed = set(eligible)
    scores = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    scores = scores[scores.index.isin(allowed)].sort_index().sort_values(ascending=False, kind="stable")
    if mode == "equal":
        raw = pd.Series(1 / len(allowed), index=sorted(allowed)) if allowed else pd.Series(dtype=float)
    elif mode == "balanced":
        fixed = {"SH510300": .6, "SH511260": .3, "SH518880": .1}
        raw = pd.Series({s: w for s, w in fixed.items() if s in allowed}, dtype=float)
    elif mode == "benchmark":
        raw = pd.Series({"SH510300": .95}) if "SH510300" in allowed else pd.Series(dtype=float)
    else:
        core = policy.get("defensive_weights", {}) if policy.get("portfolio_style") == "core_satellite" else {}
        satellite = policy.get("satellite_weight", 1.) if core else 1.
        stock_pool = allowed - set(core) - set(exiting)
        if core:
            stock_pool = {s for s in stock_pool if universe.get_asset_class(symbol_to_code(s)) == "equity"}
        scores = scores[scores.index.isin(stock_pool)]
        ranked = list(scores.index)
        held = [s for s in current.index if current[s] > 1e-6 and s in stock_pool]
        if len(scores) > 1 and scores.std() < 1e-12:
            # 无横截面区分度不根据代码排序制造选股优势；原持仓只接受风险约束。
            raw = current[current.index.isin(stock_pool)]
        else:
            # 并列按最优名次识别，不能用代码顺序在并列边界制造选股。
            ranks = scores.rank(ascending=False, method="min")
            top = set(ranks[ranks <= policy["topk"] + policy.get("rank_buffer", 0)].index)
            removable = sorted([s for s in held if s not in top], key=lambda s: (scores.get(s, -np.inf), s))[:policy["n_drop"]]
            sell = {s for s in removable if holding_days.get(s, 0) >= policy["hold_days"]}
            selected = [s for s in held if s not in sell]
            for _, group in scores[~scores.index.isin(held)].groupby(scores[~scores.index.isin(held)], sort=False):
                if len(group) > policy["topk"] - len(selected):
                    break  # 截断并列组全部留现金，不择代码买入。
                selected += list(group.index)
            raw = pd.Series(satellite / policy["topk"], index=selected) if selected else pd.Series(dtype=float)
        for symbol, weight in core.items():
            if symbol in allowed and symbol not in exiting:
                raw.loc[symbol] = weight
    desired = constrain_weights(raw, returns, policy)
    index = current.index.union(desired.index)
    old, new = current.reindex(index, fill_value=0), desired.reindex(index, fill_value=0)
    # 不阻挡风险缩减；仅使用剩余日换手额度买入。小幅漂移不机械每日调仓。
    diff = new - old
    small = diff.abs() < policy["rebalance_band"]
    small &= old.gt(0) & new.gt(0)
    # 保留旧权重只能发生在旧权重仍满足全部组合约束时。
    candidate = new.copy()
    candidate[small] = old[small]
    candidate = constrain_weights(candidate, returns, policy).reindex(index, fill_value=0)
    diff = candidate - old
    sells = -diff.clip(upper=0).sum()
    buys = diff.clip(lower=0)
    budget = max(0., policy["max_daily_turnover"] - sells)
    if buys.sum() > budget:
        candidate = old + diff.clip(upper=0) + buys * budget / buys.sum()
    return candidate[candidate > 1e-9].to_dict()


def buy_notional_budget(symbol, quantities, cash, prices, policy):
    """不预支未成交卖单；只使用实际现金和集中度余量。"""
    values = {s: q * prices[s] for s, q in quantities.items() if q > 0}
    nav = cash + sum(values.values())
    group_value = sum(v for s, v in values.items() if risk_group(s) == risk_group(symbol))
    budgets = [max(0., cash - policy["minimum_commission"]),
               nav * policy["max_weight"] - values.get(symbol, 0),
               nav * policy["max_group_weight"] - group_value,
               nav * policy["max_gross_weight"] - sum(values.values())]
    if universe.get_asset_class(symbol_to_code(symbol)) == "equity":
        equity = sum(v for s, v in values.items() if universe.get_asset_class(symbol_to_code(s)) == "equity")
        budgets.append(nav * policy["max_equity_weight"] - equity)
    return max(0., min(budgets))


def plan_orders(targets, quantities, cash, prices, adv, policy):
    """将已知价格下目标权重转为整数份额限额单；执行时仍须校验开盘与现金。"""
    prices = dict(prices)
    for symbol, quantity in quantities.items():
        if quantity > 0 and (symbol not in prices or not np.isfinite(prices[symbol]) or prices[symbol] <= 0):
            raise ValueError(f"持仓 {symbol} 缺少有效估值，停止生成订单")
    nav = float(cash) + sum(q * prices[s] for s, q in quantities.items())
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("账户净值无效")
    orders = []
    budget_cash = float(cash)
    budget_quantities = dict(quantities)
    for symbol in sorted(set(targets) | set(quantities)):
        price = prices.get(symbol, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        quantity = float(quantities.get(symbol, 0))
        desired = math.floor(nav * targets.get(symbol, 0) / price / 100) * 100
        delta = desired - quantity
        limit = float(adv.get(symbol, 0)) * policy["max_adv_participation"]
        max_qty = math.floor(max(0, limit) / price / 100) * 100
        requested = min(abs(delta), max_qty)
        if delta > 0:
            requested = math.floor(requested / 100) * 100
            if policy.get("conservative_buy_budget"):
                room = buy_notional_budget(symbol, budget_quantities, budget_cash, prices, policy)
                buffer = 1 + policy["commission"] + policy["slippage"] + policy["impact_coefficient"] * math.sqrt(policy["max_adv_participation"])
                requested = min(requested, math.floor(room / (price * buffer) / 100) * 100)
                budget_quantities[symbol] = budget_quantities.get(symbol, 0) + requested
                budget_cash -= requested * price * buffer
        if requested <= 0:
            continue
        orders.append({"symbol": symbol, "side": "buy" if delta > 0 else "sell",
                       "quantity": float(requested), "reference_price": float(price),
                       "notional_limit": float(limit), "target_weight": targets.get(symbol, 0)})
    return sorted(orders, key=lambda o: (o["side"] == "buy", o["symbol"]))
