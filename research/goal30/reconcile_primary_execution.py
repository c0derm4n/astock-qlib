"""不调用结算器，从登记规则、真实价格和成交回报独立重建每日现金与份额。"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from research.goal30.turnover_budget import require


def rounded_quantity(quantity, event):
    exact = Fraction(quantity) * Fraction(event['ratio_decimal'])
    rule = event['rounding']
    if rule == 'ceiling_to_integer_manager_funded':
        return -(-exact.numerator // exact.denominator)
    if rule == 'truncate_to_integer_remainder_to_fund':
        return exact.numerator // exact.denominator
    require(rule in ('integer_ratio_no_fraction_for_integer_holdings', 'registrar_exact_allocation_integer_ratio')
            and exact.denominator == 1, '独立核对遇到未知取整规则')
    return exact.numerator


def reconcile(frames, events, policy, scenario, reports, trades, logs, holdings, final_state, pending=None):
    dates = pd.DatetimeIndex(pd.to_datetime(reports['date']))
    require(dates.is_unique and dates.is_monotonic_increasing, '报告日期重复或倒序')
    event_by_id = {e['event_id']: e for e in events if not e.get('signal_only')}
    require(len(event_by_id) == sum(not e.get('signal_only') for e in events), '独立核对事件重复')
    schedule = {}
    uncertain_roles = {
        'custodian_outgoing_transfer_actual_broker_receipt_unconfirmed',
        'clearing_participant_transfer_actual_broker_receipt_unconfirmed',
        'announced_distribution_date_actual_broker_receipt_unconfirmed',
        'secondary_schedule_venue_not_primary_verified'}
    all_dates = [str(d.date()) for d in frames.days]
    for e in event_by_id.values():
        key = e['event_id']
        if e['type'] == 'split':
            stages = [('record_split_quantity', e['record_date']), ('split', e['effective_date'])]
        else:
            stages = [('record_entitlement', e['record_date']), ('accrue_dividend', e['ex_date'])]
            eligible_days = [d for d in all_dates if d >= e['announced_payment_date']]
            delay = scenario['payment_delay_sessions']
            delay += scenario['uncertain_payment_extra_sessions'] if e['payment_semantics'] in uncertain_roles else 0
            if delay < len(eligible_days):
                stages.append(('pay_dividend', eligible_days[delay]))
        for stage, day in stages:
            schedule.setdefault(day, []).append((stage, key))
    qty, claims, rights, split_rights = {}, {}, {}, {}
    cash = previous_nav = float(policy['initial_capital'])
    max_error, observed_rights, observed_splits, paid = 0., {}, {}, {}
    lookup = {}
    for label, frame in [('trades', trades), ('logs', logs), ('holdings', holdings)]:
        lookup[label] = {str(day): group.to_dict('records') for day, group in frame.groupby('date')} if not frame.empty else {}

    def same(a, b, label):
        nonlocal max_error
        error = abs(float(a) - float(b))
        max_error = max(max_error, error)
        require(np.isfinite(error) and error <= 1e-7, f'独立重建不符：{label}/{a}/{b}')

    for report in reports.to_dict('records'):
        day = str(report['date'])
        context = frames.context(day)
        day_trades = lookup['trades'].get(day, [])
        day_logs = lookup['logs'].get(day, [])
        expected_stages = schedule.get(day, [])
        require(len(day_logs) == len(expected_stages) and
                {(e['type'], e['event_id']) for e in day_logs} == set(expected_stages), '行动日志阶段缺失、额外或重复')
        actual_logs = {(e['type'], e['event_id']): e for e in day_logs}
        before_trades = dict(qty)
        for stage, key in expected_stages:
            e = event_by_id[key]
            if stage == 'split':
                symbol = e['symbol']
                before = qty.get(symbol, 0)
                require(before == split_rights.get(key, 0), '拆分前持仓不同于登记快照')
                after = rounded_quantity(before, e)
                if after:
                    qty[symbol] = after
                else:
                    qty.pop(symbol, None)
                observed_splits[key] = {'before': before, 'after': after}
                same(actual_logs[(stage, key)]['quantity_before'], before, '拆分前份额')
                same(actual_logs[(stage, key)]['quantity_after'], after, '拆分后份额')
        for stage, key in expected_stages:
            e = event_by_id[key]
            if stage == 'record_entitlement' and e['entitlement_basis'] == 'holdings_before_record_day_trades':
                rights[key] = before_trades.get(e['symbol'], 0) * float(e['cash_per_share_decimal'])
                require(actual_logs[(stage, key)]['phase'] == 'pre_open', '同日权益应先于交易')
        for stage, key in expected_stages:
            if stage == 'accrue_dividend':
                claims[key] = rights.get(key, 0.)
                same(actual_logs[(stage, key)]['amount'], claims[key], '除息应收')
        nominal, execution_value, charged, cost = 0., 0., 0., 0.
        action_symbols = {event_by_id[key]['symbol'] for stage, key in expected_stages if stage in ('split', 'accrue_dividend')}
        require(not action_symbols.intersection(t['symbol'] for t in day_trades), '行动日未撤销旧单位订单')
        for trade in day_trades:
            symbol, side, q = trade['symbol'], trade['side'], trade['quantity']
            price = context['open'][symbol]
            require(q > 0 and float(q).is_integer() and context['volume'][symbol] > 0 and np.isfinite(price) and price > 0, '成交缺少有效市场数据')
            require(trade['signal_date'] == context['previous_date'], '成交没有滞后一个交易日')
            if pending is not None:
                orders = pending.get(day, [])
                order = [o for o in orders if o['symbol'] == symbol and o['side'] == side]
                require(len(order) == 1 and q <= order[0]['quantity'] + 1e-10, '成交超过前日冻结订单')
                adv = order[0]['adv']
                same(trade['participation'], q * price / max(adv, 1.), '参与率')
            else:
                adv = float(frames.adv.loc[pd.Timestamp(context['previous_date']), symbol])
            slip = policy['slippage'] * scenario['slippage_multiplier'] + policy['impact_coefficient'] * math.sqrt(q * price / max(adv, 1.))
            fill = price * (1 + slip if side == 'buy' else 1 - slip)
            fee = max(policy['minimum_commission'], q * fill * policy['commission'])
            same(fill, trade['price'], '成交价格')
            same(fee, trade['commission'], '佣金')
            nominal += q * price
            execution_value += q * fill
            charge = q * max(price, fill)
            same(trade['turnover_budget_before'], previous_nav * policy['max_daily_turnover'] - charged, '成交前共享余额')
            charged += charge
            same(trade['turnover_budget_after'], previous_nav * policy['max_daily_turnover'] - charged, '成交后共享余额')
            cost += fee + q * price * slip
            if side == 'buy':
                require(q % 100 == 0 and cash >= q * fill + fee - 1e-7, '买入整手或现金约束违反')
                qty[symbol] = qty.get(symbol, 0) + int(q)
                cash -= q * fill + fee
            else:
                require(side == 'sell' and qty.get(symbol, 0) >= q, '卖出超过持仓')
                require(q % 100 == 0 or (q == qty[symbol] and q < 100), '碎股未一次清仓')
                qty[symbol] -= int(q)
                if qty[symbol] == 0:
                    qty.pop(symbol)
                cash += q * fill - fee
        require(charged <= previous_nav * policy['max_daily_turnover'] + 1e-7, '全天买卖共用预算超限')
        for stage, key in expected_stages:
            e = event_by_id[key]
            if stage == 'record_split_quantity':
                split_rights[key] = qty.get(e['symbol'], 0)
                same(actual_logs[(stage, key)]['quantity'], split_rights[key], '拆分登记快照')
            if stage == 'record_entitlement':
                if e['entitlement_basis'] == 'record_date_close_holdings':
                    rights[key] = qty.get(e['symbol'], 0) * float(e['cash_per_share_decimal'])
                    require(actual_logs[(stage, key)]['phase'] == 'post_close', '普通权益应在交易后登记')
                observed_rights[key] = rights[key]
                same(actual_logs[(stage, key)]['amount'], rights[key], '分红登记金额')
        for stage, key in expected_stages:
            if stage == 'pay_dividend':
                amount = claims.pop(key, 0.)
                cash += amount
                same(actual_logs[(stage, key)]['amount'], amount, '到账金额')
                require(actual_logs[(stage, key)]['payment_basis'] == 'announced_schedule_model', '模型支付错误标记为实际到账')
                paid[key] = amount
        values = {s: q * context['close'][s] for s, q in qty.items()}
        actual_h = lookup['holdings'].get(day, [])
        require({h['symbol'] for h in actual_h} == set(qty), '持仓标的集不符')
        for h in actual_h:
            same(h['quantity'], qty[h['symbol']], '收盘份额')
            same(h['value'], values[h['symbol']], '收盘市值')
        nav = cash + sum(values.values()) + sum(claims.values())
        for actual, expected, label in [(report['cash'], cash, '现金'), (report['value'], sum(values.values()), '市值'),
            (report['dividend_receivable'], sum(claims.values()), '应收'), (report['account'], nav, '经济净值'),
            (report['return'] - report['cost'], nav / previous_nav - 1, '净收益'),
            (report['cost'], cost / previous_nav, '成本'), (report['turnover'], nominal / previous_nav, '名义换手'),
            (report['execution_turnover'], execution_value / previous_nav, '实际换手'),
            (report['budget_turnover'], charged / previous_nav, '预算换手')]:
            same(actual, expected, label)
        previous_nav = nav
    require(qty == final_state['quantities'], '期末份额不一致')
    same(cash, final_state['cash'], '期末现金')
    require(set(claims) == set(final_state['receivable']), '期末应收集合不同')
    for key in claims:
        same(claims[key], final_state['receivable'][key], '期末应收')
    return {'days': len(dates), 'maximum_absolute_numeric_error': max_error,
        'nonzero_dividend_rights': {k: v for k, v in observed_rights.items() if v > 0},
        'nonzero_splits': {k: v for k, v in observed_splits.items() if v['before'] > 0},
        'payments': paid, 'buy_sell_budget_verified': True}
