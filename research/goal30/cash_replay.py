"""现金分红事件回放：真实ETF份额、应收权益、到账现金分别记账。"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, market30, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, OUT, SELECTED, comparison
from research.sp500.explore import allocator
from research.sp500.validate import relative_metrics
from src.portfolio import plan_orders, buy_notional_budget


def actions_from_evidence(evidence, start, end):
    events = []
    for symbol, item in evidence['symbols'].items():
        if item['comparison']['unmatched'] or item['comparison']['missing_payment_date_count']:
            raise ValueError('公司行动存在未匹配记录或缺少支付日')
        for event in item['eastmoney_events']:
            if event['type'] != 'cash_dividend' or not start <= event['ex_date'] <= end:
                continue
            if not event['record_date'] < event['ex_date'] <= event['payment_date']:
                raise ValueError('本次候选仅支持登记日早于除息日且支付日已知的分红')
            events.append({**event, 'symbol': symbol, 'event_id': symbol + '-dividend-' + event['ex_date']})
        for event in item['sina_events']:
            if event['type'] == 'split' and start <= event['effective_date'] <= end:
                events.append({**event, 'symbol': symbol, 'event_id': symbol + '-split-' + event['effective_date']})
    return events


def simulate_cash(market, candidate, policy, start, actions, *, slippage_multiplier=1., payment_delay_sessions=0):
    validate_candidate(candidate)
    if (candidate['family'] != 'fixed' or isinstance(payment_delay_sessions, bool)
            or not isinstance(payment_delay_sessions, int) or payment_delay_sessions < 0
            or not np.isfinite(slippage_multiplier) or slippage_multiplier <= 0):
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
    select = allocator(candidate, dates, start)
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
              'cash_clipped', 'risk_clipped', 'corporate_action_order_cancelled')}
    for i in range(begin, len(dates)):
        date = dates[i]
        cost = turnover = 0.
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
            capacity = math.floor(order['notional_limit'] / price / 100) * 100
            if quantity > capacity:
                issues['capacity_clipped'] += 1
                quantity = capacity
            if side == 'sell':
                quantity = min(quantity, quantities.get(symbol, 0.))
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
            trades.append({'date': str(date.date()), 'signal_date': order['signal_date'], 'symbol': symbol,
                'side': side, 'quantity': quantity, 'price': fill_price, 'commission': fee,
                'slippage_cost': quantity * price * slip, 'participation': participation})
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


def audit_cash_ledger(market, report, trades, events, initial_capital):
    cash, quantities, entitlements, claims = initial_capital, {}, {}, {}
    close = market.wide('close').ffill(limit=5)
    eligibility = market.eligibility.set_index(['date', 'symbol'])
    errors, cash_errors, claim_errors, violations = [], [], [], []
    def action(row):
        nonlocal cash
        key, symbol = row['event_id'], row['symbol']
        if row['type'] == 'split':
            if abs(quantities.get(symbol, 0.) - row['quantity_before']) > 1e-8:
                violations.append('split_quantity')
            quantities[symbol] = quantities.get(symbol, 0.) * row['ratio']
        elif row['type'] == 'record_entitlement':
            amount = quantities.get(symbol, 0.) * row['cash_per_share']
            if abs(amount - row['amount']) > .01:
                violations.append('entitlement')
            entitlements[key] = amount
        elif row['type'] == 'accrue_dividend':
            claims[key] = entitlements.get(key, 0.)
            if abs(claims[key] - row['amount']) > .01:
                violations.append('accrual')
        elif row['type'] == 'pay_dividend':
            amount = claims.pop(key, 0.)
            if abs(amount - row['amount']) > .01:
                violations.append('payment')
            cash += amount
    for date, row in report.iterrows():
        daily_events = events[events.date.eq(str(date.date()))] if not events.empty else pd.DataFrame()
        if not daily_events.empty:
            for event in daily_events[daily_events.phase.eq('pre_open')].to_dict('records'):
                action(event)
        daily_trades = trades[trades.date.eq(str(date.date()))] if not trades.empty else pd.DataFrame()
        for trade in daily_trades.to_dict('records'):
            symbol, quantity = trade['symbol'], trade['quantity']
            signal = pd.Timestamp(trade['signal_date'])
            if signal >= date or symbol not in SYMBOLS or (trade['side'] == 'buy' and
                    (quantity % 100 != 0 or not eligibility.at[(signal, symbol), 'eligible'])):
                violations.append('trade_eligibility_timing_or_lot')
            direction = 1 if trade['side'] == 'buy' else -1
            cash -= direction * quantity * trade['price'] + trade['commission']
            quantities[symbol] = quantities.get(symbol, 0.) + direction * quantity
            if cash < -.01 or quantities[symbol] < -1e-8:
                violations.append('negative_cash_or_quantity')
        if not daily_events.empty:
            for event in daily_events[daily_events.phase.eq('post_close')].to_dict('records'):
                action(event)
        value = sum(q * close.at[date, s] for s, q in quantities.items() if q > 1e-8)
        cash_errors.append(abs(cash - row.cash))
        claim_errors.append(abs(sum(claims.values()) - row.dividend_receivable))
        errors.append(abs(cash + value + sum(claims.values()) - row.account))
    result = {'rows': len(report), 'trades': len(trades), 'actions': len(events),
              'max_nav_error': max(errors), 'max_cash_error': max(cash_errors),
              'max_receivable_error': max(claim_errors), 'violations': sorted(set(violations)),
              'maximum_adv_participation': float(trades.participation.max()) if not trades.empty else 0.}
    if max(errors + cash_errors + claim_errors) > .01 or violations or result['maximum_adv_participation'] > .02 + 1e-10:
        raise ValueError(f'独立现金账本校验失败：{result}')
    return result


def run():
    protocol_path = Path(__file__).with_name('cash_replay_protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    selected, _ = load_selected(SELECTED)
    source = BASE / protocol['actions_source']
    evidence = json.loads(source.read_text(encoding='utf-8'))
    required = set(selected['candidate']['weights']) | {s for a in selected['candidate'].get('fallbacks', {}).values() for s in a}
    if required - set(evidence['symbols']):
        raise ValueError('交易标的缺少公司行动资料')
    actions = actions_from_evidence(evidence, protocol['start'], protocol['end'])
    market = market30(BASE / selected['source'])
    if str(market.calendar[-1].date()) != protocol['end']:
        raise ValueError('行情截止日与登记不一致')
    directory = OUT / 'cash_replay' / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    paths = [source, protocol_path, Path(__file__)]
    for item in evidence['symbols'].values():
        for item_source in item['sources']:
            paths.append(BASE / item_source['archive'] if 'archive' in item_source else source.parent / item_source['file'])
    write_json_atomic(directory / 'frozen_protocol.json', {'protocol': protocol, 'candidate': selected['candidate'],
        'policy': selected['policy'], 'actions': actions, 'input_hashes': {**selected['hashes'],
            **{p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}})
    result = {'cases': {}, 'corporate_actions_audited': False, 'historical_universe_audited': False,
              'investment_admission': False, 'limitations': protocol['limitations']}
    for case in protocol['cases']:
        name = case['name']
        report, trades, event_log, holdings, issues = simulate_cash(market, selected['candidate'], selected['policy'],
            protocol['start'], actions, **{k: v for k, v in case.items() if k != 'name'})
        _, benchmark = relative_metrics(report, BASE / selected['benchmark'], selected['comparison'])
        result['cases'][name] = {**comparison(report.account, benchmark.spy_total_cny), 'issues': issues,
            'ledger': audit_cash_ledger(market, report, trades, event_log, selected['policy']['initial_capital']),
            'cash_dividends_received': float(event_log.loc[event_log.type.eq('pay_dividend'), 'amount'].sum()),
            'max_dividend_receivable': float(report.dividend_receivable.max())}
        for prefix, frame in [('report', report), ('trades', trades), ('actions', event_log), ('holdings', holdings)]:
            frame.to_csv(directory / f'{prefix}_{name}.csv', index=prefix == 'report')
        print(json.dumps({'case': name, 'metrics': result['cases'][name]['strategy'],
                          'checks': result['cases'][name]['checks']}, ensure_ascii=True), flush=True)
    load_selected(SELECTED)
    write_json_atomic(directory / 'results.json', result)
    write_json_atomic(OUT / 'cash_replay/latest.json', {'directory': str(directory)})
    print(str(directory), flush=True)
    return directory


if __name__ == '__main__':
    run()
