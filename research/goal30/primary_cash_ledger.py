"""原件行动结算层；复用冻结成交/共享预算代码，不把模型支付称为实际到账。"""
from __future__ import annotations

import copy
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN

import numpy as np
import pandas as pd

from research.ex_nasdaq.core import SYMBOLS
from research.goal30 import daily_cash_ledger as execution
from research.goal30.daily_cash_ledger import fingerprint, seal
from research.goal30.turnover_budget import require


def shares_after_split(quantity, event):
    q, ratio = Decimal(str(quantity)), Decimal(event['ratio_decimal'])
    require(q.is_finite() and q >= 0 and q == q.to_integral_value(), '场内拆分前份额必须为非负整数')
    require(ratio.is_finite() and ratio > 0, '拆分比例无效')
    exact = q * ratio
    rule = event['rounding']
    if rule == 'ceiling_to_integer_manager_funded':
        after = exact.to_integral_value(rounding=ROUND_CEILING)
    elif rule == 'truncate_to_integer_remainder_to_fund':
        after = exact.to_integral_value(rounding=ROUND_DOWN)
    else:
        require(rule in ('integer_ratio_no_fraction_for_integer_holdings', 'registrar_exact_allocation_integer_ratio')
                and ratio == ratio.to_integral_value() and exact == exact.to_integral_value(),
                '缺少适用的份额取整证据')
        after = exact
    return int(after), str(exact), str(after - exact)


def validate_state(state):
    execution.validate_state(state)
    require(state.get('schema_version') == 2 and state.get('ledger_variant') == 'primary_action_settlement_v1',
            '须使用新行动账本，禁止静默迁移旧账户')
    for quantity in list(state['quantities'].values()) + list(state['split_entitled'].values()):
        q = Decimal(str(quantity))
        require(q.is_finite() and q >= 0 and q == q.to_integral_value(), '场内账本份额非整数')
    require(isinstance(state['receipt_ids'], list) and len(set(state['receipt_ids'])) == len(state['receipt_ids']), '到账回报ID重复')


def new_state(initial_cash, previous_date, *, mode='historical_validation'):
    state = execution.new_state(initial_cash, previous_date, mode=mode)
    state.update(schema_version=2, ledger_variant='primary_action_settlement_v1',
                 split_entitled={}, receipt_ids=[], settlement_policy_sha256=None)
    return seal(state)


class EventCalendar:
    def __init__(self, actions, days, *, payment_mode='confirmed_only', payment_delay_sessions=0,
                 uncertain_payment_extra_sessions=1):
        require(payment_mode in ('confirmed_only', 'announced_schedule_model'), '支付模式无效')
        for value in (payment_delay_sessions, uncertain_payment_extra_sessions):
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, '支付延迟须为非负整数')
        self.days = pd.DatetimeIndex(days)
        require(len(self.days) > 1 and self.days.is_unique and self.days.is_monotonic_increasing, '日历无效')
        self.payment_mode = payment_mode
        self.policy_hash = fingerprint({'payment_mode': payment_mode, 'payment_delay_sessions': payment_delay_sessions,
                                        'uncertain_payment_extra_sessions': uncertain_payment_extra_sessions})
        self.events, self.schedule = {}, {}
        for source in actions:
            if source.get('signal_only'):
                continue
            event = copy.deepcopy(source)
            key = event['event_id']
            require(key not in self.events and event['symbol'] in SYMBOLS, '行动重复或标的在池外')
            if event['type'] == 'split':
                require(event['record_date'] <= event['book_date'] < event['effective_date'], '拆分日期顺序不符')
                shares_after_split(1, event)  # 未知规则不能只因暂时空仓而放过。
                stages = [('split_record', event['record_date']), ('split', event['effective_date'])]
            else:
                require(event['type'] == 'cash_dividend', '不支持的公司行动')
                cash = Decimal(event['cash_per_share_decimal'])
                require(cash.is_finite() and cash > 0, '分红金额无效')
                require(event['record_date'] <= event['ex_date'] <= event['announced_payment_date'], '分红日期顺序不符')
                if event['record_date'] == event['ex_date']:
                    require(event['entitlement_basis'] == 'holdings_before_record_day_trades', '同日分红缺少交易前权益依据')
                    record_stage = 'record_pre'
                else:
                    require(event['entitlement_basis'] == 'record_date_close_holdings', '普通分红权益依据不符')
                    record_stage = 'record_close'
                stages = [(record_stage, event['record_date']), ('ex', event['ex_date'])]
                if payment_mode == 'announced_schedule_model':
                    uncertain = event['payment_semantics'] in (
                        'custodian_outgoing_transfer_actual_broker_receipt_unconfirmed',
                        'clearing_participant_transfer_actual_broker_receipt_unconfirmed',
                        'announced_distribution_date_actual_broker_receipt_unconfirmed',
                        'secondary_schedule_venue_not_primary_verified')
                    index = self.days.searchsorted(pd.Timestamp(event['announced_payment_date']))
                    index += payment_delay_sessions + (uncertain_payment_extra_sessions if uncertain else 0)
                    if index < len(self.days):
                        stages.append(('pay_model', str(self.days[index].date())))
            self.events[key] = event
            for stage, day in stages:
                timestamp = pd.Timestamp(day)
                if self.days[0] <= timestamp <= self.days[-1]:
                    require(timestamp in self.days, '行动阶段日期缺少真实交易日')
                self.schedule.setdefault(str(timestamp.date()), {}).setdefault(stage, []).append(event)
        self.hashes = {key: fingerprint(event) for key, event in self.events.items()}

    def for_day(self, day):
        row = self.schedule.get(str(pd.Timestamp(day).date()), {})
        return {stage: row.get(stage, []) for stage in ('split_record', 'split', 'record_pre', 'record_close', 'ex', 'pay_model')}

    def check(self, state, context):
        i = self.days.get_loc(pd.Timestamp(context['date']))
        require(i > 0 and str(self.days[i - 1].date()) == context['previous_date'] == state['last_date'], '账本漏日或日历衔接不符')
        if self.payment_mode == 'announced_schedule_model':
            require(state['mode'] == 'historical_validation', '公告支付模型只用于历史开发，不能冒充每日真实到账')
        require(state['settlement_policy_sha256'] in (None, self.policy_hash), '不能中途改变支付模型')
        for key, expected in state['event_fingerprints'].items():
            require(self.hashes.get(key) == expected, '既有事件被修改或删除')
        for key in self.hashes.keys() - state['event_fingerprints'].keys():
            require(state['sequence'] == 0 or self.events[key]['record_date'] > state['last_date'], '新增事件涉及已结算权益日')


class _NoExecutionActions:
    """公司行动由本层处理；冻结成交器只处理原有撮合、费用和共享预算。"""
    @staticmethod
    def check_updates(state):
        return state['event_fingerprints']

    @staticmethod
    def for_day(day):
        return {key: [] for key in ('splits', 'ex', 'record', 'pay')}


def entitlement(state, event, phase, day):
    quantity = state['quantities'].get(event['symbol'], 0)
    amount = float(Decimal(str(quantity)) * Decimal(event['cash_per_share_decimal']))
    state['entitled'][event['event_id']] = amount
    return {'date': day, 'phase': phase, 'type': 'record_entitlement', 'event_id': event['event_id'],
        'symbol': event['symbol'], 'quantity': quantity, 'cash_per_share': float(event['cash_per_share_decimal']),
        'amount': amount, 'entitlement_basis': event['entitlement_basis']}


def settle_day(original, context, policy, catalog, *, slippage_multiplier=1., receipts=()):
    validate_state(original)
    catalog.check(original, context)
    state = copy.deepcopy(original)
    day = context['date']
    events = catalog.for_day(day)
    stages = ['primary:' + e['event_id'] + ':' + phase for phase, items in events.items() for e in items]
    require(not set(stages).intersection(state['processed_stages']), '行动阶段已经处理')
    state['event_fingerprints'] = catalog.hashes.copy()
    state['settlement_policy_sha256'] = catalog.policy_hash
    if state['sequence'] == 0:
        for event in catalog.events.values():
            if event['record_date'] <= state['last_date']:
                require(not state['quantities'].get(event['symbol'], 0), '初始已有持仓缺少历史登记权益，不能假设为零')
                if event['type'] == 'split':
                    state['split_entitled'].setdefault(event['event_id'], 0)
                else:
                    state['entitled'].setdefault(event['event_id'], 0.)
                    if event['ex_date'] <= state['last_date']:
                        state['receivable'].setdefault(event['event_id'], 0.)
    logs = []
    for event in events['split']:
        key, symbol = event['event_id'], event['symbol']
        before = state['quantities'].get(symbol, 0)
        require(key in state['split_entitled'] or (state['sequence'] == 0 and before == 0), '缺少拆分登记快照')
        require(before == state['split_entitled'].get(key, 0), '拆分登记后份额改变，需核对停牌/交易与登记记录')
        after, exact, difference = shares_after_split(before, event)
        if after:
            state['quantities'][symbol] = after
        else:
            state['quantities'].pop(symbol, None)
            state['bought'].pop(symbol, None)
            state['exiting'] = [s for s in state['exiting'] if s != symbol]
        logs.append({'date': day, 'phase': 'pre_open', 'type': 'split', 'event_id': key, 'symbol': symbol,
            'ratio': float(event['ratio_decimal']), 'quantity_before': before, 'quantity_after': after,
            'quantity_exact_decimal': exact, 'rounding_difference_shares_decimal': difference,
            'rounding_rule': event['rounding']})
    for event in events['record_pre']:
        logs.append(entitlement(state, event, 'pre_open', day))
    for event in events['ex']:
        key = event['event_id']
        require(key in state['entitled'] or (state['sequence'] == 0 and not state['quantities']), '缺少分红登记快照')
        amount = state['entitled'].get(key, 0.)
        state['receivable'][key] = amount
        logs.append({'date': day, 'phase': 'pre_open', 'type': 'accrue_dividend',
            'event_id': key, 'symbol': event['symbol'], 'amount': amount})
    cancelled_symbols = {e['symbol'] for e in events['split'] + events['ex']}
    cancelled = sum(order['symbol'] in cancelled_symbols for order in state['pending_orders'])
    state['pending_orders'] = [order for order in state['pending_orders'] if order['symbol'] not in cancelled_symbols]
    # 继续沿用保守规则：折算/除息日撤销昨收单位下的旧订单，当晚重新计算。
    state, output = execution.settle_day(seal(state), context, policy, _NoExecutionActions(),
                                        slippage_multiplier=slippage_multiplier)
    output['issues']['corporate_action_order_cancelled'] += cancelled
    for event in events['split_record']:
        quantity = state['quantities'].get(event['symbol'], 0)
        state['split_entitled'][event['event_id']] = quantity
        logs.append({'date': day, 'phase': 'post_close', 'type': 'record_split_quantity',
            'event_id': event['event_id'], 'symbol': event['symbol'], 'quantity': quantity})
    for event in events['record_close']:
        logs.append(entitlement(state, event, 'post_close', day))
    payments = []
    if catalog.payment_mode == 'announced_schedule_model':
        require(not receipts, '历史模型支付不能同时注入券商回报')
        payments = [(e, state['receivable'].get(e['event_id']), 'model:' + e['event_id'], 'announced_schedule_model')
                    for e in events['pay_model']]
    else:
        for receipt in receipts:
            require(receipt['date'] == day and receipt.get('receipt_id') and receipt.get('evidence_reference'), '到账回报缺少当天日期或证据引用')
            require(receipt['event_id'] in catalog.events, '到账对应未知事件')
            event = catalog.events[receipt['event_id']]
            require(event['type'] == 'cash_dividend' and day >= event['ex_date'], '分红回报日期早于除息')
            payments.append((event, receipt['amount'], receipt['receipt_id'], 'caller_supplied_receipt'))
    settlement_difference = 0.
    for event, amount, receipt_id, basis in payments:
        key = event['event_id']
        require(receipt_id not in state['receipt_ids'] and key in state['receivable'], '重复到账或缺少应收权益')
        expected = state['receivable'][key]
        require(amount is not None and np.isfinite(amount) and amount >= 0 and abs(amount - expected) <= .005000001,
                '到账金额与应收不符，须先核对税费或分配差异')
        settlement_difference += amount - expected
        state['receivable'].pop(key)
        state['cash'] += float(amount)
        state['receipt_ids'].append(receipt_id)
        logs.append({'date': day, 'phase': 'post_close', 'type': 'pay_dividend', 'event_id': key,
            'symbol': event['symbol'], 'amount': amount, 'expected_receivable': expected,
            'settlement_difference': amount - expected, 'receipt_id': receipt_id, 'payment_basis': basis})
    report = output['report']
    report['account'] += settlement_difference
    report['return'] += settlement_difference / original['previous_nav']
    report['cash'] = state['cash']
    report['dividend_receivable'] = sum(state['receivable'].values())
    report['dividend_settlement_difference'] = settlement_difference
    state['previous_nav'] = report['account']
    state['processed_stages'] += stages
    validate_state(seal(state))  # 先拒绝非整数，不能用int转换掩盖意外的小数持仓。
    state['quantities'] = {s: int(q) for s, q in state['quantities'].items()}
    output['actions'] = logs
    state = seal(state)
    validate_state(state)
    return state, output


def freeze_next(state, targets, close_prices, adv, policy, execution_date):
    validate_state(state)
    result = execution.freeze_next(state, targets, close_prices, adv, policy, execution_date)
    validate_state(result)
    return result
