"""冻结现金引擎的独立执行扩展；新增单日双边成交额硬预算。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from research.ex_nasdaq.core import SYMBOLS, validate_candidate
from research.sp500.explore import allocator
from src.portfolio import plan_orders, buy_notional_budget
from research.goal30.turnover_budget import DailyTurnoverBudget, quote, round_quantity


def simulate_budgeted_cash(market, candidate, policy, start, actions, *, slippage_multiplier=1.,
                           payment_delay_sessions=0, select_factory=allocator, enforce_turnover=True):
    validate_candidate(candidate)
    if (candidate['family'] != 'fixed' or isinstance(payment_delay_sessions, bool)
            or not isinstance(payment_delay_sessions, int) or payment_delay_sessions < 0
            or not np.isfinite(slippage_multiplier) or slippage_multiplier <= 0
            or not isinstance(enforce_turnover, bool)):
        raise ValueError('现金回放仅用于冻结固定权重候选，支付延迟必须非负')
    close, opened, volume = (market.wide(k) for k in ('close', 'open', 'volume'))
    close_mark = close.ffill(limit=5)
    returns = market.wide('adj_close').pct_change(fill_method=None)
    eligible = market.eligibility.pivot(index='date', columns='symbol', values='eligible')
    adv = market.eligibility.pivot(index='date', columns='symbol', values='avg_amount')
    dates = market.calendar
    begin = max(int(dates.searchsorted(pd.Timestamp(start))), 1)
    if begin >= len(dates):
        raise ValueError('回放起点没有可用交易日')
    select = select_factory(candidate, dates, start)
    events, record_events, ex_events, pay_events, splits = {}, {}, {}, {}, {}
    for event in actions:
        key = event['event_id']
        if key in events or event['symbol'] not in SYMBOLS:
            raise ValueError('公司行动ID重复或标的不在30只池中')
        events[key] = event
        if event['type'] == 'split':
            splits.setdefault(pd.Timestamp(event['effective_date']), []).append(event)
        else:
            record_events.setdefault(pd.Timestamp(event['record_date']), []).append(event)
            ex_events.setdefault(pd.Timestamp(event['ex_date']), []).append(event)
            payment = dates.searchsorted(pd.Timestamp(event['payment_date'])) + payment_delay_sessions
            if payment < len(dates):
                pay_events.setdefault(dates[payment], []).append(event)
    cash = previous_nav = float(policy['initial_capital'])
    quantities, bought, entitled, receivable = {}, {}, {}, {}
    exiting, queue, records, trades, actions_log, holdings = set(), {}, [], [], [], []
    issues = {key: 0 for key in ('suspended_or_missing', 'price_limit', 'capacity_clipped',
              'cash_clipped', 'risk_clipped', 'corporate_action_order_cancelled', 'turnover_clipped')}
    for i in range(begin, len(dates)):
        date = dates[i]
        cost = turnover = execution_notional = 0.
        budget = DailyTurnoverBudget(previous_nav, policy['max_daily_turnover']) if enforce_turnover else None
        action_symbols = set()
        for event in splits.get(date, []):
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
        for event in ex_events.get(date, []):
            amount = entitled.get(event['event_id'], 0.)
            receivable[event['event_id']] = amount
            action_symbols.add(event['symbol'])
            actions_log.append({'date': str(date.date()), 'phase': 'pre_open', 'type': 'accrue_dividend',
                'event_id': event['event_id'], 'symbol': event['symbol'], 'amount': amount})
        for order in queue.pop(i, []):
            symbol, side = order['symbol'], order['side']
            if symbol in action_symbols:
                issues['corporate_action_order_cancelled'] += 1
                continue
            price = opened.at[date, symbol]
            if not np.isfinite(price) or price <= 0 or volume.at[date, symbol] <= 0:
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
                    prices = close_mark.iloc[i-1].to_dict()
                    prices.update(opened.loc[date].dropna().to_dict())
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
        for event in record_events.get(date, []):
            quantity = quantities.get(event['symbol'], 0.)
            amount = quantity * event['cash_per_share']
            entitled[event['event_id']] = amount
            actions_log.append({'date': str(date.date()), 'phase': 'post_close', 'type': 'record_entitlement',
                'event_id': event['event_id'], 'symbol': event['symbol'], 'quantity': quantity,
                'cash_per_share': event['cash_per_share'], 'amount': amount})
        for event in pay_events.get(date, []):
            amount = receivable.pop(event['event_id'], 0.)
            cash += amount
            actions_log.append({'date': str(date.date()), 'phase': 'post_close', 'type': 'pay_dividend',
                'event_id': event['event_id'], 'symbol': event['symbol'], 'amount': amount})
        values = {}
        for symbol, quantity in quantities.items():
            price = close_mark.at[date, symbol]
            if not np.isfinite(price):
                raise ValueError('持仓连续缺价，不能虚构估值')
            values[symbol] = quantity * price
            holdings.append({'date': date, 'symbol': symbol, 'quantity': quantity, 'value': values[symbol]})
        liquid_nav = cash + sum(values.values())
        claims = sum(receivable.values())
        nav = liquid_nav + claims
        if cash < -1e-6 or liquid_nav <= 0 or nav <= 0:
            raise ValueError('现金账户余额或净值无效')
        net = nav / previous_nav - 1
        records.append({'date': date, 'account': nav, 'return': net + cost / previous_nav,
                        'cost': cost / previous_nav, 'turnover': turnover / previous_nav,
                        'cash': cash, 'value': sum(values.values()), 'dividend_receivable': claims})
        if budget is not None:
            records[-1].update({'execution_turnover': execution_notional / previous_nav,
                               'budget_turnover': budget.spent / previous_nav,
                               'turnover_budget_limit': budget.limit, 'turnover_budget_remaining': budget.remaining})
        if i + 1 < len(dates):
            allowed = eligible.columns[eligible.loc[date].fillna(False)].tolist()
            targets = select(pd.Series(dtype=float), allowed, {s: v / liquid_nav for s, v in values.items()},
                             {s: i - bought[s] + 1 for s in quantities}, returns.loc[:date], policy, exiting=exiting)
            orders = plan_orders(targets, quantities, cash, close_mark.loc[date], adv.loc[date], policy)
            for order in orders:
                order.update({'signal_date': str(date.date()), 'adv': float(adv.at[date, order['symbol']])})
            queue.setdefault(i+1, []).extend(orders)
        previous_nav = nav
    report = pd.DataFrame(records).set_index('date')
    report['bench'] = market.wide('adj_close')['SH510300'].pct_change(fill_method=None).reindex(report.index).fillna(0.)
    return report, pd.DataFrame(trades), pd.DataFrame(actions_log), pd.DataFrame(holdings), issues
