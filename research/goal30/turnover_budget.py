"""买卖共享的单日成交额硬预算；报价与成交数量一致。"""
from __future__ import annotations

import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quote(quantity, side, reference_price, adv, policy, slippage_multiplier):
    require(side in ('buy', 'sell'), '成交方向无效')
    require(all(math.isfinite(float(x)) for x in (quantity, reference_price, adv, slippage_multiplier))
            and quantity >= 0 and reference_price > 0 and adv > 0 and slippage_multiplier > 0,
            '报价输入必须有限且价格、成交额、滑点倍数为正')
    require(all(math.isfinite(float(policy[k])) and policy[k] >= 0 for k in ('slippage', 'impact_coefficient')),
            '滑点和冲击参数无效')
    participation = quantity * reference_price / adv
    slip = policy['slippage'] * slippage_multiplier + policy['impact_coefficient'] * math.sqrt(participation)
    require(slip < 1, '报价滑点达到或超过100%')
    fill_price = reference_price * (1 + slip if side == 'buy' else 1 - slip)
    return participation, slip, fill_price


def round_quantity(quantity, odd_exit=False):
    """只有整笔清空不足一手的余量才允许碎股；其他买卖按100份向下取整。"""
    require(math.isfinite(float(quantity)) and quantity >= 0, '数量无效')
    return quantity if odd_exit and quantity < 100 else math.floor(quantity / 100) * 100


class DailyTurnoverBudget:
    def __init__(self, previous_nav, fraction):
        require(math.isfinite(float(previous_nav)) and previous_nav > 0, '预算净值必须为正')
        require(not isinstance(fraction, bool) and math.isfinite(float(fraction)) and 0 < fraction <= 1,
                '换手比例须位于0至1之间')
        self.limit = float(previous_nav * fraction)
        self.spent = 0.

    @property
    def remaining(self):
        return max(0., self.limit - self.spent)

    @staticmethod
    def charge(quantity, reference_price, fill_price):
        # 买入滑点增加成交额；卖出则以较大的开盘名义金额预留，两个口径都不会突破预算。
        require(all(math.isfinite(float(x)) for x in (quantity, reference_price, fill_price))
                and quantity >= 0 and reference_price > 0 and fill_price > 0, '预算扣减输入无效')
        return quantity * max(reference_price, fill_price)

    def cap_quantity(self, requested, side, reference_price, adv, policy, slippage_multiplier, *, odd_exit=False):
        requested = round_quantity(requested, odd_exit and side == 'sell')

        def fits(quantity):
            _, _, fill_price = quote(quantity, side, reference_price, adv, policy, slippage_multiplier)
            return self.charge(quantity, reference_price, fill_price) <= self.remaining

        if requested < 100:
            return requested if fits(requested) else 0.
        low, high = 0, int(requested // 100)
        # 买入成交额随份额单调增加；卖出的保守预算扣减为份额乘开盘价。
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle * 100):
                low = middle
            else:
                high = middle - 1
        return float(low * 100)

    def commit(self, quantity, reference_price, fill_price):
        charge = self.charge(quantity, reference_price, fill_price)
        require(charge <= self.remaining + 1e-8, '实际成交超过剩余预算')
        self.spent += charge
        require(self.spent <= self.limit + 1e-8, '实际成交超过全天预算')
        return charge
