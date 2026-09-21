"""30只候选的真实前瞻模拟记录：冻结账户信号、核对成交、累计连续观察日。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from filelock import FileLock

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS
from research.goal30.audit import OUT


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _now():
    return pd.Timestamp.now(tz='Asia/Shanghai')


def _time(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError('前瞻记录时间必须携带时区')
    return value.tz_convert('Asia/Shanghai')


def runtime_fingerprint():
    files = [Path(__file__), Path(__file__).with_name('daily.py')]
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def scope(selected, runtime_hash, account_path, root=None):
    account_key = hashlib.sha256(str(Path(account_path).resolve()).encode()).hexdigest()[:24]
    # Windows默认路径长度有限；完整指纹仍保留在冻结决策中。
    return Path(root or OUT / 'forward') / f'{fingerprint(selected)[:24]}-{runtime_hash[:24]}' / account_key


def paper_decision(observation, account, account_path):
    if observation['executable'] or observation['orders']:
        raise ValueError('前瞻入口只接收显式模拟观察')
    identity = fingerprint({'observation_id': observation['observation_id'], 'account': account,
                            'account_path': str(Path(account_path).resolve()),
                            'orders': observation['paper_orders']})[:24]
    orders = [{**o, 'order_id': f'{identity}-{i+1}'} for i, o in enumerate(observation['paper_orders'])]
    return {'schema_version': 2, 'decision_id': identity, 'mode': 'paper', 'executable': False,
            'account_revision': account['revision'], 'account_fingerprint': fingerprint(account),
            'signal_date': observation['signal_date'], 'execution_date': observation['execution_date'],
            'execution_window': 'next_session_open', 'orders': orders,
            'target_weights': observation['target_weights'], 'account_nav': observation['account_nav'],
            'strategy_fingerprint': observation['strategy_fingerprint'],
            'runtime_sha256': observation['runtime_sha256'], 'created_at': observation['checked_at']}


def freeze(directory, decision, account):
    now = _now()
    signal = pd.Timestamp(decision['signal_date'])
    execution = pd.Timestamp(decision['execution_date'])
    calendar = xcals.get_calendar('XSHG')
    if decision.get('mode') != 'paper' or decision.get('executable') is not False:
        raise ValueError('只能冻结明确不可实盘执行的模拟决策')
    if (str(now.date()) != decision['signal_date'] or now.strftime('%H:%M') < '19:00'
            or not calendar.is_session(signal) or calendar.next_session(signal) != execution
            or _time(decision['created_at']) > now or _time(decision['created_at']).date() != now.date()
            or _time(decision['created_at']).strftime('%H:%M') < '19:00'):
        raise ValueError('只能在信号日19:00后冻结下一交易日信号，禁止历史补记')
    if fingerprint(account) != decision['account_fingerprint']:
        raise ValueError('冻结账户与决策账户不一致')
    path = Path(directory) / 'signals' / f'{execution.date()}.json'
    payload = {'frozen_at': now.isoformat(), 'decision': decision, 'opening_account': account}
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['decision'] != decision or old['opening_account'] != account:
                raise ValueError('执行日已有冻结信号，禁止事后更换账户或订单')
        else:
            write_json_atomic(path, payload)
    return path


def reconcile(opening, closing, decision, now, min_fill_ratio=.95):
    """独立核对账户现金/份额及每笔回报；账户增删改不能冒充策略收益。"""
    reasons = []
    cash = float(opening['cash'])
    quantities = {s: float(h['quantity']) for s, h in opening['holdings'].items()}
    for account in (opening, closing):
        if (not np.isfinite(account['cash']) or account['cash'] < 0 or set(account['holdings']) - SYMBOLS
                or any(not np.isfinite(h['quantity']) or h['quantity'] <= 0 for h in account['holdings'].values())):
            raise ValueError('前瞻账户现金或份额无效')
    orders = {o['order_id']: o for o in decision['orders']}
    if len(orders) != len(decision['orders']) or any(
            o['symbol'] not in SYMBOLS or o['side'] not in ('buy', 'sell')
            or not np.isfinite(o['quantity']) or o['quantity'] <= 0 for o in orders.values()):
        raise ValueError('前瞻冻结订单无效')
    completed = {key: 0. for key in orders}
    for field in ('executions', 'corporate_actions'):
        if any(closing.get(field, {}).get(k) != v for k, v in opening.get(field, {}).items()):
            reasons.append(f'旧{field}记录被修改或删除')
    actions = [(k, v) for k, v in closing.get('corporate_actions', {}).items()
               if k not in opening.get('corporate_actions', {})]
    for _, row in actions:
        event = row['event']
        signature = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        if (signature != row['fingerprint'] or event['effective_date'] != decision['execution_date']
                or not event.get('source') or event.get('symbol') not in SYMBOLS):
            reasons.append('公司行动来源、日期或指纹无效')
            continue
        symbol = event['symbol']
        if event['type'] == 'split' and symbol in quantities:
            ratio = float(event['ratio'])
            if not np.isfinite(ratio) or ratio <= 0:
                reasons.append('折算比例无效')
            else:
                quantities[symbol] *= ratio
        elif event['type'] in ('cash_dividend', 'delist_settlement'):
            amount = float(event['cash_amount'])
            if not np.isfinite(amount) or amount < 0:
                reasons.append('公司行动现金无效')
            else:
                cash += amount
                if event['type'] == 'delist_settlement':
                    quantities.pop(symbol, None)
        else:
            reasons.append('无法核对公司行动')
    new_fills = [v for k, v in closing['executions'].items() if k not in opening['executions']]
    locked_buys = {}
    for row in sorted(new_fills, key=lambda v: v['fill']['executed_at']):
        fill = row['fill']
        signature = hashlib.sha256(json.dumps(fill, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        order = orders.get(fill.get('order_id'))
        if (signature != row['fingerprint'] or order is None or fill.get('decision_id') != decision['decision_id']
                or fill.get('symbol') != order['symbol'] or fill.get('side') != order['side']):
            reasons.append('存在不属于冻结决策或指纹异常的成交')
            continue
        time = _time(fill['executed_at'])
        if (str(time.date()) != decision['execution_date'] or time > now
                or not '09:25' <= time.strftime('%H:%M') <= '09:35'):
            reasons.append('成交不在计划开盘窗口或时间在未来')
        q, price, fee = (float(fill[k]) for k in ('quantity', 'price', 'fee'))
        if not all(np.isfinite(v) for v in (q, price, fee)) or q <= 0 or price <= 0 or fee < 0:
            reasons.append('成交数量、价格或费用无效')
            continue
        symbol = fill['symbol']
        completed[fill['order_id']] += q
        if completed[fill['order_id']] > order['quantity'] + 1e-8:
            reasons.append('成交超过冻结订单份额')
        if fill['side'] == 'buy':
            if q != int(q):
                reasons.append('买入成交非整数份额')
            cash -= q * price + fee
            quantities[symbol] = quantities.get(symbol, 0.) + q
            locked_buys[symbol] = locked_buys.get(symbol, 0.) + q
        else:
            if q > quantities.get(symbol, 0.) - locked_buys.get(symbol, 0.) + 1e-8:
                reasons.append('超卖或违反保守T+1')
            cash += q * price - fee
            quantities[symbol] = quantities.get(symbol, 0.) - q
        if cash < -.01:
            reasons.append('成交路径现金为负')
    ratios = [completed[k] / o['quantity'] for k, o in orders.items()]
    minimum = min(ratios, default=1.)
    if minimum < min_fill_ratio:
        reasons.append('订单成交完成率不足95%')
    cash_error = abs(cash - closing['cash'])
    actual_quantities = {s: float(h['quantity']) for s, h in closing['holdings'].items()}
    quantity_error = max((abs(quantities.get(s, 0.) - actual_quantities.get(s, 0.))
                          for s in set(quantities) | set(actual_quantities)), default=0.)
    if cash_error > .01 or quantity_error > 1e-8:
        reasons.append('账户现金或份额无法由冻结账户与回报重建')
    return {'execution_aligned': not reasons, 'reasons': sorted(set(reasons)),
            'minimum_order_fill_ratio': minimum, 'cash_error': cash_error, 'quantity_error': quantity_error,
            'new_fill_count': len(new_fills), 'new_corporate_action_count': len(actions)}


def record_close(directory, account, market, min_fill_ratio=.95):
    now = _now()
    day = str(now.date())
    if now.strftime('%H:%M') < '19:00' or str(market.calendar[-1].date()) != day:
        raise ValueError('只能在当天正式收盘数据齐备后记录，禁止历史补记')
    signal_path = Path(directory) / 'signals' / f'{day}.json'
    if not signal_path.exists():
        return {'status': 'no_prior_signal', 'qualified': False}
    frozen = json.loads(signal_path.read_text(encoding='utf-8'))
    decision = frozen['decision']
    cutoff = now.normalize() + pd.Timedelta(hours=9, minutes=25)
    if (_time(frozen['frozen_at']) >= cutoff or decision['execution_date'] != day
            or fingerprint(frozen['opening_account']) != decision['account_fingerprint']):
        raise ValueError('信号未在执行前冻结')
    if set(account['holdings']) - SYMBOLS:
        raise ValueError('前瞻账户含池外标的')
    prices = market.bars[market.bars.date == pd.Timestamp(day)].set_index('symbol').close.to_dict()
    if any(s not in prices or not np.isfinite(prices[s]) or prices[s] <= 0 for s in account['holdings']):
        raise ValueError('前瞻持仓缺少当天有效价格')
    nav = account['cash'] + sum(h['quantity'] * prices[s] for s, h in account['holdings'].items())
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError('前瞻账户净值无效')
    audit = reconcile(frozen['opening_account'], account, decision, now, min_fill_ratio)
    payload = {'date': day, 'recorded_at': now.isoformat(), 'nav': nav, 'closing_account': account,
               'opening_reference_nav': decision['account_nav'], 'decision_id': decision['decision_id'],
               'signal_sha256': hashlib.sha256(signal_path.read_bytes()).hexdigest(),
               'data_fingerprint': market.fingerprint, 'execution_audit': audit,
               'closing_prices': {s: prices[s] for s in account['holdings']}}
    path = Path(directory) / 'closes' / f'{day}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['closing_account'] != account or old['nav'] != nav:
                raise ValueError('前瞻收盘已记录且账户或行情变化，须人工核对，禁止覆盖')
        else:
            write_json_atomic(path, payload)
    return {'status': 'recorded', 'qualified': audit['execution_aligned'], 'path': str(path),
            'reasons': audit['reasons']}


def evidence(directory):
    now = _now()
    calendar = xcals.get_calendar('XSHG')
    rows, invalid = [], []
    streak = 0
    previous = None
    for path in sorted((Path(directory) / 'closes').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        signal_path = Path(directory) / 'signals' / f"{row['date']}.json"
        if not signal_path.exists():
            invalid.append({'date': row['date'], 'reason': 'missing_signal'})
            streak = 0
            previous = None
            continue
        frozen = json.loads(signal_path.read_text(encoding='utf-8'))
        decision = frozen['decision']
        recorded, frozen_at = _time(row['recorded_at']), _time(frozen['frozen_at'])
        day = pd.Timestamp(row['date'])
        time_ok = (str(recorded.date()) == row['date'] and recorded.strftime('%H:%M') >= '19:00'
                   and recorded <= now and calendar.is_session(day)
                   and str(frozen_at.date()) == decision['signal_date'] and frozen_at.strftime('%H:%M') >= '19:00'
                   and calendar.is_session(pd.Timestamp(decision['signal_date']))
                   and calendar.next_session(pd.Timestamp(decision['signal_date'])) == day
                   and frozen_at < recorded.normalize() + pd.Timedelta(hours=9, minutes=25))
        identity_ok = (hashlib.sha256(signal_path.read_bytes()).hexdigest() == row['signal_sha256']
                       and fingerprint(frozen['opening_account']) == decision['account_fingerprint']
                       and decision['decision_id'] == row['decision_id'] and decision['execution_date'] == row['date'])
        audit = reconcile(frozen['opening_account'], row['closing_account'], decision, recorded)
        recalculated = row['closing_account']['cash'] + sum(
            h['quantity'] * row['closing_prices'][s] for s, h in row['closing_account']['holdings'].items())
        value_ok = (np.isfinite(recalculated) and recalculated > 0 and np.isfinite(row['nav'])
                    and abs(recalculated - row['nav']) <= .01
                    and abs(row['opening_reference_nav'] - decision['account_nav']) <= .01)
        valid = time_ok and identity_ok and value_ok and audit['execution_aligned'] and audit == row['execution_audit']
        continuous = (previous is None or (calendar.next_session(pd.Timestamp(previous['date'])) == day
                      and fingerprint(previous['closing_account']) == fingerprint(frozen['opening_account'])
                      and abs(previous['nav'] - row['opening_reference_nav']) <= .01))
        streak = (streak + 1 if continuous else 1) if valid else 0
        if not valid:
            invalid.append({'date': row['date'], 'reason': 'time_identity_or_ledger_failed'})
        previous = row
        rows.append(row)
    today = now.tz_localize(None).normalize()
    expected = calendar.date_to_session(today, direction='previous')
    if calendar.is_session(today) and now.strftime('%H:%M') < '19:00':
        expected = calendar.previous_session(today)
    fresh = bool(rows) and rows[-1]['date'] == str(expected.date())
    return {'recorded_days': len(rows), 'qualified_consecutive_days': streak,
            'required_days': 126, 'fresh': fresh, 'forward_gate_met': streak >= 126 and fresh,
            'last_recorded_date': rows[-1]['date'] if rows else None, 'invalid_days': invalid,
            'note': '真实日期的模拟账户与回报验收；不会由观察文件数量或历史回填推算合格天数'}
