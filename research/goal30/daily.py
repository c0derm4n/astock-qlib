"""30只ETF冻结候选的每日收盘观察；实盘准入未通过时不生成可执行指令。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from filelock import FileLock

import config
from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, OUT, SELECTED
from research.goal30 import forward
from research.sp500.explore import allocator
from src.account import read_account
from src.market import load_market
from src.portfolio import plan_orders


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def session_state(now):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('检查时间必须携带时区')
    now = now.tz_convert('Asia/Shanghai')
    day = now.tz_localize(None).normalize()
    calendar = xcals.get_calendar('XSHG')
    if not calendar.is_session(day):
        return {'status': 'market_closed', 'date': str(day.date()),
                'next_session': str(calendar.date_to_session(day, direction='next').date())}
    if now.strftime('%H:%M') < config.DECISION_RUN_TIME:
        return {'status': 'not_due', 'date': str(day.date()), 'run_after': config.DECISION_RUN_TIME}
    return {'status': 'due', 'date': str(day.date()),
            'execution_date': str(calendar.next_session(day).date())}


def observe(selected, market, state, account=None):
    """纯计算入口；仅使用当天可知信息，供真实日更与独立诊断共同调用。"""
    if state['status'] != 'due':
        raise ValueError('只能使用正式收盘后的交易日')
    candidate, policy = selected['candidate'], selected['policy']
    validate_candidate(candidate)
    if candidate['family'] != 'fixed' or candidate.get('execution_window'):
        raise ValueError('本每日入口仅支持已冻结的每日固定权重候选')
    if set(selected['symbols']) != SYMBOLS:
        raise ValueError('冻结策略必须限定于指定30只ETF')
    date = pd.Timestamp(state['date'])
    if market.calendar[-1] != date:
        raise ValueError('当日正式行情未齐备，拒绝使用过期数据生成信号')
    bars = market.bars[market.bars.symbol.isin(SYMBOLS)].copy()
    eligibility = market.eligibility[market.eligibility.symbol.isin(SYMBOLS)]
    latest = bars[bars.date == date].set_index('symbol')
    eligible_today = eligibility[eligibility.date == date].set_index('symbol')
    allowed = set(eligible_today.index[eligible_today.eligible]) & set(latest.index)
    required = set(candidate['weights'])
    for alternatives in candidate.get('fallbacks', {}).values():
        required.update(alternatives)
    if required - set(latest.index):
        raise ValueError(f'配置标的缺少当日正式价格：{sorted(required - set(latest.index))}')
    prices = latest.close.to_dict()
    if any(not np.isfinite(p) or p <= 0 for p in prices.values()):
        raise ValueError('当日价格无效')
    raw = {s: w for s, w in candidate['weights'].items() if s in allowed}
    substitutes = []
    for original, alternatives in candidate.get('fallbacks', {}).items():
        if original not in allowed:
            for symbol, fraction in alternatives.items():
                if symbol in allowed:
                    raw[symbol] = raw.get(symbol, 0.) + candidate['weights'][original] * fraction
                    substitutes.append({'from': original, 'to': symbol})
    blockers = ['历史目录与公司行动审计未完成', '冻结规则真实前瞻尚未通过126日验收',
                '原投资准入回撤门槛未通过；结果依赖2025年通信行情']
    result = {'schema_version': 1, 'status': 'research_only', 'executable': False,
              'signal_date': state['date'], 'execution_date': state['execution_date'],
              'execution_window': 'next_session_open', 'candidate': candidate['name'],
              'allowed_symbols': sorted(SYMBOLS), 'eligible_symbols': sorted(allowed),
              'allocation_reference': raw, 'cash_reference_weight': 1 - sum(raw.values()),
              'eligibility_substitutions': substitutes, 'reference_prices': {s: prices[s] for s in sorted(required)},
              'rebalance_band': policy['rebalance_band'], 'rebalance_needed': None,
              'orders': [], 'paper_orders': [], 'blockers': blockers,
              'data_fingerprint': market.fingerprint,
              'investment_admission': False, 'forward_admission_days': 0,
              'forward_note': '权重观察不等同于有核对账户和成交回报的合格前瞻日'}
    if account is None:
        result['blockers'].append('未提供核对账户；不能判断实际持仓漂移或计算份额')
        return result
    quantities = {s: h['quantity'] for s, h in account['holdings'].items()}
    if set(quantities) - SYMBOLS:
        raise ValueError('账户存在30只池外持仓，须人工核对后再进行本策略观察')
    for symbol, holding in account['holdings'].items():
        if symbol not in prices or pd.Timestamp(holding['buy_date']) > date:
            raise ValueError('账户持仓日期或当日估值不完整')
    nav = account['cash'] + sum(q * prices[s] for s, q in quantities.items())
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError('账户净值无效')
    current = {s: q * prices[s] / nav for s, q in quantities.items()}
    history = bars.pivot(index='date', columns='symbol', values='adj_close').reindex(market.calendar)
    returns = history.pct_change(fill_method=None).loc[:date]
    # 固定候选每天检查，无排名状态；复用回测的资格替代、漂移和分批建仓分配器。
    targets = allocator(candidate, market.calendar, date)(pd.Series(dtype=float), allowed, current, {},
               returns, policy, exiting=[s for s, h in account['holdings'].items() if h.get('exit_pending')])
    paper_orders = plan_orders(targets, quantities, account['cash'], prices,
                               eligible_today.avg_amount.to_dict(), policy)
    if any(order['symbol'] not in SYMBOLS for order in paper_orders):
        raise ValueError('模拟计划出现池外订单')
    result.update({'account_revision': account['revision'], 'account_nav': nav,
                   'current_weights': current, 'target_weights': targets,
                   'rebalance_needed': bool(paper_orders), 'paper_orders': paper_orders,
                   'paper_note': '以下仅为未通过准入的模拟测算，不可作为实盘指令'})
    return result


def run(*, account_path=None, refresh=False, output_dir=None, now=None):
    now = pd.Timestamp.now(tz='Asia/Shanghai') if now is None else pd.Timestamp(now)
    directory = Path(output_dir or OUT / 'daily')
    directory.mkdir(parents=True, exist_ok=True)
    state = session_state(now)
    if state['status'] != 'due':
        payload = {**state, 'executable': False, 'orders': [], 'notification_needed': False,
                   'checked_at': now.isoformat()}
        write_json_atomic(directory / 'latest_check.json', payload)
        return payload
    with FileLock(str(directory / 'daily.lock')):
        try:
            selected, _ = load_selected(SELECTED)
            runtime_hash = forward.runtime_fingerprint()
            account_key = str(Path(account_path).resolve()) if account_path else 'weights_only'
            identity = digest({'selected': selected, 'runtime': runtime_hash, 'account': account_key})[:24]
            destination = directory / identity / f"{state['date']}.json"
            if destination.exists():
                payload = json.loads(destination.read_text(encoding='utf-8'))
                if account_path:
                    with FileLock(str(account_path) + '.lock'):
                        current_account = read_account(account_path)
                    if digest(current_account) != payload.get('account_fingerprint'):
                        raise ValueError('当日信号冻结后账户已变化，须核对原记录，禁止覆盖冻结建议')
                return {**payload, 'already_recorded': True, 'notification_needed': False}
            if refresh:
                log_path = directory / f"refresh_{state['date']}.log"
                with log_path.open('a', encoding='utf-8') as log:
                    process = subprocess.run([sys.executable, '-m', 'src.fetch_data'], cwd=BASE,
                                             stdout=log, stderr=subprocess.STDOUT, timeout=900)
                if process.returncode:
                    raise ValueError(f'正式行情更新失败，详见 {log_path}')
            with patch.object(config, 'EXCLUDE_QDII', False):
                market = load_market(as_of=state['date'])
            account = None
            if account_path:
                with FileLock(str(account_path) + '.lock'):
                    account = read_account(account_path)
            payload = observe(selected, market, state, account)
            payload.update({'checked_at': now.isoformat(), 'runtime_sha256': runtime_hash,
                            'strategy_fingerprint': digest(selected), 'observation_id': identity + '-' + state['date']})
            if account is not None:
                forward_directory = forward.scope(selected, runtime_hash, account_path,
                    root=directory / 'forward' if output_dir else None)
                payload['account_fingerprint'] = digest(account)
                payload['forward_close'] = forward.record_close(forward_directory, account, market,
                                                               selected['policy']['min_forward_fill_ratio'])
                decision = forward.paper_decision(payload, account, account_path)
                signal_path = forward.freeze(forward_directory, decision, account)
                decision_path = directory / identity / 'paper_decisions' / f"{state['date']}.json"
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(decision_path, decision)
                payload['paper_decision_path'] = str(decision_path)
                payload['forward_signal_path'] = str(signal_path)
                payload['forward_evidence'] = forward.evidence(forward_directory)
                payload['forward_admission_days'] = payload['forward_evidence']['qualified_consecutive_days']
                payload['forward_note'] = payload['forward_evidence']['note']
                if payload['forward_evidence']['forward_gate_met']:
                    payload['blockers'].remove('冻结规则真实前瞻尚未通过126日验收')
                if payload['forward_close']['status'] == 'recorded' and not payload['forward_close']['qualified']:
                    payload['blockers'].append('当日前瞻未通过：' + '；'.join(payload['forward_close']['reasons']))
            key = digest({k: payload[k] for k in ('status', 'blockers', 'allocation_reference',
                                                'eligibility_substitutions', 'paper_orders')})
            previous = directory / identity / 'last_action_key.json'
            old = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
            payload['notification_needed'] = old.get('key') != key
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(destination, payload)
            write_json_atomic(previous, {'key': key})
            payload['observation_path'] = str(destination)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            payload = {'status': 'data_or_configuration_error', 'executable': False, 'orders': [],
                       'signal_date': state['date'], 'checked_at': now.isoformat(), 'error': str(exc),
                       'notification_needed': True}
        write_json_atomic(directory / 'latest_check.json', payload)
        return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=Path, help='可选，用户已核对的模拟/人工账户；省略则仅观察权重')
    parser.add_argument('--refresh', action='store_true', help='交易日19:00后先更新正式行情，再检查一次')
    args = parser.parse_args()
    result = run(account_path=args.account, refresh=args.refresh)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] == 'data_or_configuration_error':
        raise SystemExit(1)
