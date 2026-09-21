"""可恢复的逐日模拟账本；新入口，不改冻结批量执行器。"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd

from research.ex_nasdaq.core import SYMBOLS
from research.goal30.turnover_budget import DailyTurnoverBudget, quote, require
from src.portfolio import plan_orders, buy_notional_budget


ISSUES = ('suspended_or_missing', 'price_limit', 'capacity_clipped', 'cash_clipped',
          'risk_clipped', 'corporate_action_order_cancelled', 'turnover_clipped')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def seal(state):
    state = copy.deepcopy(state)
    state.pop('state_sha256', None)
    state['state_sha256'] = fingerprint(state)
    return state


def validate_state(state):
    value = {k: v for k, v in state.items() if k != 'state_sha256'}
    require(state.get('state_sha256') == fingerprint(value), '账本状态指纹不一致')
    require(state.get('mode') in ('historical_validation', 'paper_model')
            and state.get('executable') is False, '账本只能用于明确标记的模拟')
    require(np.isfinite(state['cash']) and state['cash'] >= -1e-6
            and np.isfinite(state['previous_nav']) and state['previous_nav'] > 0,
            '现金或经济净值无效')
    require(set(state['quantities']).issubset(SYMBOLS)
            and all(np.isfinite(q) and q > 0 for q in state['quantities'].values()), '持仓无效或在池外')
    for field in ('entitled', 'receivable'):
        require(all(np.isfinite(x) and x >= 0 for x in state[field].values()), '分红权益无效')
    require(set(state['bought']) == set(state['quantities'])
            and set(state['exiting']).issubset(state['quantities']), '持仓和退出状态不一致')


def new_state(initial_cash, previous_date, *, mode='historical_validation'):
    require(np.isfinite(initial_cash) and initial_cash > 0, '初始模拟现金必须为正')
    state = seal({'schema_version': 1, 'mode': mode, 'executable': False,
        'last_date': str(pd.Timestamp(previous_date).date()), 'sequence': 0,
        'cash': float(initial_cash), 'previous_nav': float(initial_cash),
        'quantities': {}, 'bought': {}, 'exiting': [], 'entitled': {}, 'receivable': {},
        'pending_orders': [], 'event_fingerprints': {}, 'processed_stages': [],
        'next_execution_date': None})
    validate_state(state)
    return state


class EventCalendar:
    """支付延迟仅供历史压力测试；目录不可静默修改或补入已处理的事件。"""
    def __init__(self, actions, calendar, payment_delay_sessions=0):
        require(isinstance(payment_delay_sessions, int) and not isinstance(payment_delay_sessions, bool)
                and payment_delay_sessions >= 0, '支付延迟须为非负整数')
        self.days = pd.DatetimeIndex(calendar)
        require(self.days.is_unique and self.days.is_monotonic_increasing, '事件交易日历无效')
        self.events, self.schedule = {}, {}
        for action in actions:
            event = copy.deepcopy(action)
            key = event['event_id']
            require(key not in self.events and event['symbol'] in SYMBOLS, '事件ID重复或标的在池外')
            if event['type'] == 'split':
                require(np.isfinite(event['ratio']) and event['ratio'] > 0, '折算比例无效')
                stages = [('splits', event['effective_date'])]
            else:
                require(event['type'] == 'cash_dividend' and np.isfinite(event['cash_per_share'])
                        and event['cash_per_share'] >= 0, '公司行动类型或分红金额无效')
                record, ex, pay = (pd.Timestamp(event[k]) for k in ('record_date', 'ex_date', 'payment_date'))
                require(record < ex <= pay, '分红日期顺序无效')
                stages = [('record', record), ('ex', ex)]
                pos = self.days.searchsorted(pay) + payment_delay_sessions
                if pos < len(self.days):
                    stages.append(('pay', self.days[pos]))
            self.events[key] = event
            for stage, day in stages:
                day = pd.Timestamp(day)
                if self.days[0] <= day <= self.days[-1]:
                    require(day in self.days, '登记、除息或拆分日期不是交易日')
                self.schedule.setdefault(str(day.date()), {}).setdefault(stage, []).append(event)

    def for_day(self, day):
        value = self.schedule.get(str(pd.Timestamp(day).date()), {})
        return {key: value.get(key, []) for key in ('splits', 'ex', 'record', 'pay')}

    def check_updates(self, state):
        hashes = {key: fingerprint(event) for key, event in self.events.items()}
        for key, expected in state['event_fingerprints'].items():
            require(hashes.get(key) == expected, '既有公司行动被删除或修改')
        for key in hashes.keys() - state['event_fingerprints'].keys():
            event = self.events[key]
            first = event.get('record_date', event.get('effective_date'))
            # 初始现金账户没有历史持仓权益；首日允许载入历史目录。
            require(state['sequence'] == 0 or first > state['last_date'], '新增事件涉及已结算日期，须人工核对')
        return hashes


def settle_day(original, context, policy, catalog, *, slippage_multiplier=1.):
    """仅执行前日已保存的订单；收盘信号由freeze_next另行计算。"""
    validate_state(original)
    require(np.isfinite(slippage_multiplier) and slippage_multiplier > 0, '滑点倍数无效')
    state = copy.deepcopy(original)
    date = pd.Timestamp(context['date'])
    require(context['previous_date'] == state['last_date'] and date > pd.Timestamp(state['last_date']),
            '账本日期重复、倒序或中断')
    if state['next_execution_date'] is not None:
        require(state['next_execution_date'] == context['date'], '执行日期与前日冻结日期不符')
    state['event_fingerprints'] = catalog.check_updates(state)
    day_events = catalog.for_day(date)
    stages = [event['event_id'] + ':' + stage for stage, events in day_events.items() for event in events]
    require(not set(stages).intersection(state['processed_stages']), '公司行动阶段重复处理')
    for order in state['pending_orders']:
        require(order['signal_date'] == state['last_date'] and order['execution_date'] == context['date']
                and order['symbol'] in SYMBOLS and order['side'] in ('buy', 'sell')
                and all(np.isfinite(order[k]) and order[k] > 0
                        for k in ('quantity', 'reference_price', 'notional_limit', 'adv')), '冻结订单无效')
    cash, previous_nav = state['cash'], state['previous_nav']
    quantities, bought, entitled, receivable = (state[k] for k in ('quantities', 'bought', 'entitled', 'receivable'))
    exiting = set(state['exiting'])
    i = state['sequence'] + 1
    records, trades, actions_log, holdings = [], [], [], []
    issues = {key: 0 for key in ISSUES}

    cost = turnover = execution_notional = 0.
    budget = DailyTurnoverBudget(previous_nav, policy['max_daily_turnover'])
    action_symbols = set()
    for event in day_events['splits']:
        symbol = event['symbol']
        ratio = float(event['ratio'])
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('折算比例无效')
        before = quantities.get(symbol, 0.)
        if before:
            quantities[symbol] = before * ratio
        action_symbols.add(symbol)
        actions_log.append({'date': str(date.date()), 'phase': 'pre_open', 'type': 'split',
            'event_id': event['event_id'], 'symbol': symbol, 'ratio': ratio,
            'quantity_before': before, 'quantity_after': before * ratio})
    for event in day_events['ex']:
        amount = entitled.get(event['event_id'], 0.)
        receivable[event['event_id']] = amount
        action_symbols.add(event['symbol'])
        actions_log.append({'date': str(date.date()), 'phase': 'pre_open', 'type': 'accrue_dividend',
            'event_id': event['event_id'], 'symbol': event['symbol'], 'amount': amount})
    for order in state['pending_orders']:
        symbol, side = order['symbol'], order['side']
        if symbol in action_symbols:
            issues['corporate_action_order_cancelled'] += 1
            continue
        price = context['open'].get(symbol, float('nan'))
        if not np.isfinite(price) or price <= 0 or not np.isfinite(context['volume'].get(symbol, np.nan)) or context['volume'].get(symbol, 0.) <= 0:
            issues['suspended_or_missing'] += 1
            continue
        threshold = .195 if symbol in ('SH588000', 'SZ159915') else .095
        if abs(price / order['reference_price'] - 1) >= threshold:
            issues['price_limit'] += 1
            continue
        quantity = math.floor(order['quantity'] / 100) * 100
        if side == 'sell' and order['target_weight'] == 0 and order['quantity'] < 100:
            quantity = order['quantity']
        odd_exit = side == 'sell' and order['target_weight'] == 0 and order['quantity'] < 100
        capacity = math.floor(order['notional_limit'] / price / 100) * 100
        if budget is not None and odd_exit:
            capacity = order['notional_limit'] / price
        if quantity > capacity:
            issues['capacity_clipped'] += 1
            quantity = (0. if budget is not None and odd_exit else capacity)
        if side == 'sell':
            quantity = min(quantity, quantities.get(symbol, 0.))
        if quantity <= 0:
            continue
        if budget is not None:
            limited = budget.cap_quantity(quantity, side, price, order['adv'], policy,
                                          slippage_multiplier, odd_exit=odd_exit)
            if limited < quantity:
                issues['turnover_clipped'] += 1
            quantity = limited
            if quantity <= 0:
                continue
        participation = quantity * price / max(order['adv'], 1.)
        slip = policy['slippage'] * slippage_multiplier + policy['impact_coefficient'] * math.sqrt(participation)
        fill_price = price * (1 + slip if side == 'buy' else 1 - slip)
        if side == 'buy':
            if policy.get('conservative_buy_budget'):
                prices = dict(context['previous_close'])
                prices.update({s: p for s, p in context['open'].items() if np.isfinite(p)})
                room = buy_notional_budget(symbol, quantities, cash, prices, policy)
                cap = math.floor(room / (fill_price * (1 + policy['commission'])) / 100) * 100
                if quantity > cap:
                    issues['risk_clipped'] += 1
                    quantity = cap
            affordable = math.floor(max(0., cash - policy['minimum_commission']) /
                         (fill_price * (1 + policy['commission'])) / 100) * 100
            if quantity > affordable:
                issues['cash_clipped'] += 1
                quantity = affordable
        if quantity <= 0:
            continue
        if budget is not None:
            # 现金/集中度限制后再次以最终实际数量报价，避免沿用较大订单的参与率。
            participation, slip, fill_price = quote(quantity, side, price, order['adv'], policy, slippage_multiplier)
            before = budget.remaining
            charge = budget.commit(quantity, price, fill_price)
        fee = max(policy['minimum_commission'], quantity * fill_price * policy['commission'])
        direction = 1 if side == 'buy' else -1
        cash -= direction * quantity * fill_price + fee
        if side == 'buy' and quantities.get(symbol, 0.) <= 1e-9:
            bought[symbol] = i
        quantities[symbol] = quantities.get(symbol, 0.) + direction * quantity
        if quantities[symbol] <= 1e-8:
            quantities.pop(symbol, None)
            bought.pop(symbol, None)
            exiting.discard(symbol)
        elif side == 'sell' and order['target_weight'] <= 1e-9:
            exiting.add(symbol)
        cost += fee + quantity * price * slip
        turnover += quantity * price
        execution_notional += quantity * fill_price
        trades.append({'date': str(date.date()), 'signal_date': order['signal_date'], 'symbol': symbol,
            'side': side, 'quantity': quantity, 'price': fill_price, 'commission': fee,
            'slippage_cost': quantity * price * slip, 'participation': participation})
        if budget is not None:
            trades[-1].update({'raw_price': price, 'execution_notional': quantity * fill_price,
                'turnover_budget_before': before, 'turnover_budget_charge': charge,
                'turnover_budget_after': budget.remaining})
    for event in day_events['record']:
        quantity = quantities.get(event['symbol'], 0.)
        amount = quantity * event['cash_per_share']
        entitled[event['event_id']] = amount
        actions_log.append({'date': str(date.date()), 'phase': 'post_close', 'type': 'record_entitlement',
            'event_id': event['event_id'], 'symbol': event['symbol'], 'quantity': quantity,
            'cash_per_share': event['cash_per_share'], 'amount': amount})
    for event in day_events['pay']:
        amount = receivable.pop(event['event_id'], 0.)
        cash += amount
        actions_log.append({'date': str(date.date()), 'phase': 'post_close', 'type': 'pay_dividend',
            'event_id': event['event_id'], 'symbol': event['symbol'], 'amount': amount})
    values = {}
    for symbol, quantity in quantities.items():
        price = context['close'].get(symbol, float('nan'))
        if not np.isfinite(price):
            raise ValueError('持仓连续缺价，不能虚构估值')
        values[symbol] = quantity * price
        holdings.append({'date': str(date.date()), 'symbol': symbol, 'quantity': quantity, 'value': values[symbol]})
    liquid_nav = cash + sum(values.values())
    claims = sum(receivable.values())
    nav = liquid_nav + claims
    if cash < -1e-6 or liquid_nav <= 0 or nav <= 0:
        raise ValueError('现金账户余额或净值无效')
    net = nav / previous_nav - 1
    records.append({'date': str(date.date()), 'account': nav, 'return': net + cost / previous_nav,
                    'cost': cost / previous_nav, 'turnover': turnover / previous_nav,
                    'cash': cash, 'value': sum(values.values()), 'dividend_receivable': claims})
    if budget is not None:
        records[-1].update({'execution_turnover': execution_notional / previous_nav,
                           'budget_turnover': budget.spent / previous_nav,
                               'turnover_budget_limit': budget.limit, 'turnover_budget_remaining': budget.remaining})
    state.update({'cash': cash, 'previous_nav': nav, 'last_date': str(date.date()),
                  'sequence': i, 'exiting': sorted(exiting), 'pending_orders': [],
                  'next_execution_date': None,
                  'processed_stages': state['processed_stages'] + stages})
    state = seal(state)
    validate_state(state)
    return state, {'report': records[0], 'trades': trades, 'actions': actions_log,
                   'holdings': holdings, 'issues': issues}


def freeze_next(original, targets, close_prices, adv, policy, execution_date):
    validate_state(original)
    require(original['next_execution_date'] is None, '次日信号已经冻结')
    require(pd.Timestamp(execution_date) > pd.Timestamp(original['last_date']), '次日信号日期无效')
    require(set(targets).issubset(SYMBOLS) and all(np.isfinite(w) and 0 <= w <= policy['max_weight'] + 1e-10
            for w in targets.values()) and sum(targets.values()) <= policy['max_gross_weight'] + 1e-10,
            '目标权重无效或违反30只范围及仓位上限')
    state = copy.deepcopy(original)
    # 资金分配与冻结回放保持一致：应收计经济净值，但不当作可买入资金。
    orders = plan_orders(targets, state['quantities'], state['cash'], close_prices, adv, policy)
    for order in orders:
        require(np.isfinite(adv[order['symbol']]) and adv[order['symbol']] > 0, '信号日成交额无效')
        order.update({'signal_date': state['last_date'], 'execution_date': str(pd.Timestamp(execution_date).date()),
                      'adv': float(adv[order['symbol']])})
    state['pending_orders'] = orders
    state['next_execution_date'] = str(pd.Timestamp(execution_date).date())
    return seal(state)
