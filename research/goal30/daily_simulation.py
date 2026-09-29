"""独立每日模型模拟：持久保存应收与共享预算；不计真实前瞻合格日。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from filelock import FileLock

import config
from src.artifacts import json_safe, write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, validate_candidate
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, SELECTED
from research.goal30.daily import session_state
from research.goal30.daily_cash_ledger import (
    EventCalendar, fingerprint, freeze_next, new_state, require, settle_day, validate_state)
from research.sp500.explore import allocator
from src.market import load_market


class MarketDays:
    def __init__(self, market):
        self.market = market
        self.days = pd.DatetimeIndex(market.calendar)
        require(self.days.is_unique and self.days.is_monotonic_increasing, '行情日期无效')
        self.close = market.wide('close').ffill(limit=5)
        self.open = market.wide('open')
        self.volume = market.wide('volume')
        self.adv = market.eligibility.pivot(index='date', columns='symbol', values='avg_amount')
        self.eligible = market.eligibility.pivot(index='date', columns='symbol', values='eligible')
        self.returns = market.wide('adj_close').pct_change(fill_method=None)

    def context(self, date):
        day = pd.Timestamp(date)
        i = self.days.get_loc(day)
        require(i > 0, '缺少前一交易日行情')
        return {'date': str(day.date()), 'previous_date': str(self.days[i-1].date()),
            'open': self.open.loc[day].to_dict(), 'close': self.close.loc[day].to_dict(),
            'previous_close': self.close.iloc[i-1].to_dict(), 'volume': self.volume.loc[day].to_dict()}

    def fixed_targets(self, selected, state, start):
        day = pd.Timestamp(state['last_date'])
        allowed = self.eligible.columns[self.eligible.loc[day].fillna(False)].tolist()
        values = {s: q * self.close.at[day, s] for s, q in state['quantities'].items()}
        liquid_nav = state['cash'] + sum(values.values())
        # 固定配置的原始目标恒定；以当前日初始化，恢复时不依赖旧闭包raw。
        # 原批量分配器在初始/月末设定同一固定目标，然后每天按资格和漂移调整。
        require(selected['candidate']['family'] == 'fixed' and not selected['candidate'].get('execution_window'),
                '无状态重建仅适用于每日固定配置')
        select = allocator(selected['candidate'], self.days, day)
        return select(pd.Series(dtype=float), allowed, {s: v/liquid_nav for s, v in values.items()},
            {s: state['sequence']-state['bought'][s]+1 for s in values}, self.returns.loc[:day],
            selected['policy'], exiting=state['exiting'])


def read_store(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    signature = payload.pop('store_sha256', None)
    require(signature == fingerprint(payload), '持久账本内容或指纹被修改')
    require(payload['mode'] == 'paper_model' and payload['sessions'], '每日模拟存档无效')
    previous = None
    for row in payload['sessions']:
        validate_state(row['state'])
        require(row['state']['mode'] == 'paper_model' and row['date'] == row['state']['last_date'], '记录日期或模式不符')
        if previous:
            require(row['parent_state_sha256'] == previous['state']['state_sha256']
                    and row['date'] == previous['state']['next_execution_date'], '账本逐日衔接中断')
        previous = row
    return payload


def save_store(path, payload):
    require('store_sha256' not in payload, '禁止把旧存档指纹用于新状态')
    write_json_atomic(Path(path), {**payload, 'store_sha256': fingerprint(payload)})


def runtime_fingerprint():
    paths = ['research/goal30/daily_simulation.py', 'research/goal30/daily_cash_ledger.py',
             'research/goal30/turnover_budget.py', 'research/goal30/daily.py',
             'research/sp500/explore.py', 'src/portfolio.py', 'src/market.py',
             'research/ex_nasdaq/core.py', 'config.py', 'universe.py']
    return fingerprint({p: hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths})


def run_model_day(path, selected, market, event_feed, *, initial_cash=None, now=None):
    """now仅用于离线测试；CLI总是读取真实系统时间。不接受用户历史日期参数。"""
    now = pd.Timestamp.now(tz='Asia/Shanghai') if now is None else pd.Timestamp(now)
    status = session_state(now)
    if status['status'] != 'due':
        return {**status, 'executable': False, 'orders': [], 'qualified_forward_days': 0}
    day, next_day = status['date'], status['execution_date']
    require(str(market.calendar[-1].date()) == day, '正式行情未更新到当天，拒绝补记或过期观察')
    require(event_feed['as_of'] == day and event_feed['covered_through'] == day
            and isinstance(event_feed.get('source_note'), str) and event_feed['source_note'].strip(),
            '公司行动输入须说明来源与当天覆盖日期；自报覆盖不等于完整目录审计')
    validate_candidate(selected['candidate'])
    require(selected['candidate']['family'] == 'fixed' and not selected['candidate'].get('execution_window')
            and set(selected['symbols']) == SYMBOLS, '本入口只支持冻结的每日固定配置与30只池')
    policy = selected['policy']
    frames = MarketDays(market)
    context = frames.context(day)
    required = set(selected['candidate']['weights'])
    for alternatives in selected['candidate'].get('fallbacks', {}).values():
        required.update(alternatives)
    today = market.bars[market.bars.date == pd.Timestamp(day)].set_index('symbol')
    require(required.issubset(today.index) and all(np.isfinite(today.at[s, 'close'])
            and today.at[s, 'close'] > 0 for s in required), '配置标的缺少当日正式价格')
    calendar = xcals.get_calendar('XSHG')
    require(str(calendar.previous_session(pd.Timestamp(day)).date()) == context['previous_date'], '行情缺少前一真实交易日')
    event_days = frames.days.union(pd.DatetimeIndex([pd.Timestamp(next_day)]))
    catalog = EventCalendar(event_feed['events'], event_days)
    identity = fingerprint({'strategy': selected, 'runtime': runtime_fingerprint()})
    inputs = fingerprint(json_safe({'market_fingerprint': market.fingerprint,
        'context': context, 'events': event_feed, 'identity': identity}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        if path.exists():
            store = read_store(path)
            require(store['identity'] == identity, '策略或执行代码变更，禁止续写旧模拟账本')
            previous = store['sessions'][-1]
            if previous['date'] == day:
                require(previous['inputs_sha256'] == inputs, '当日冻结后输入变化，拒绝覆盖')
                require(initial_cash is None or initial_cash == store['initial_cash'], '不能通过重试变更初始资金')
                return {**previous['observation'], 'already_recorded': True}
            require(initial_cash is None, '已建账后不能重新注入初始资金')
            opening = previous['state']
            require(opening['next_execution_date'] == day, '每日模拟漏跑，须核对后另建账；禁止历史补记')
        else:
            require(initial_cash is not None, '首次建模拟账须显式指定初始现金')
            opening = new_state(initial_cash, context['previous_date'], mode='paper_model')
            store = {'schema_version': 1, 'mode': 'paper_model', 'identity': identity,
                     'initial_cash': float(initial_cash), 'start': day, 'sessions': []}
        state, output = settle_day(opening, context, policy, catalog)
        targets = frames.fixed_targets(selected, state, store['start'])
        closed = copy.deepcopy(state)
        state = freeze_next(state, targets, context['close'], frames.adv.loc[pd.Timestamp(day)].to_dict(), policy, next_day)
        observation = {'status': 'paper_model_recorded', 'executable': False, 'orders': [],
            'signal_date': day, 'execution_date': next_day, 'recorded_at': now.isoformat(),
            'economic_nav': output['report']['account'], 'cash': state['cash'],
            'dividend_receivable': sum(state['receivable'].values()),
            'liquid_allocation_nav': output['report']['account'] - output['report']['dividend_receivable'],
            'target_weights': targets, 'paper_orders': state['pending_orders'],
            'execution_turnover': output['report']['execution_turnover'],
            'turnover_budget_remaining': output['report']['turnover_budget_remaining'],
            'rebalance_needed': bool(state['pending_orders']), 'qualified_forward_days': 0,
            'investment_admission': False,
            'note': '成交为开盘价与成本模型的模拟，非券商回报；不计入126日真实前瞻。数据完整性及收益回撤准入仍未通过。'}
        store['sessions'].append({'date': day, 'inputs_sha256': inputs,
            'parent_state_sha256': opening['state_sha256'], 'state': state, 'closing_state_sha256': closed['state_sha256'],
            'events_as_of': event_feed['as_of'], 'event_source_note': event_feed['source_note'],
            'output': output, 'observation': observation})
        save_store(path, store)
        return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True, help='独立模拟账本路径')
    parser.add_argument('--events', type=Path, required=True, help='当天公司行动目录，含as_of/covered_through/source_note/events')
    parser.add_argument('--initial-cash', type=float, help='仅首次建模时指定，不代表用户真实账户')
    args = parser.parse_args()
    now = pd.Timestamp.now(tz='Asia/Shanghai')
    status = session_state(now)
    if status['status'] != 'due':
        print(json.dumps({**status, 'executable': False, 'orders': [], 'qualified_forward_days': 0}, ensure_ascii=False))
        return
    selected, _ = load_selected(SELECTED)
    with patch.object(config, 'EXCLUDE_QDII', False):
        market = load_market(as_of=status['date'])
    feed = json.loads(args.events.read_text(encoding='utf-8'))
    result = run_model_day(args.ledger, selected, market, feed, initial_cash=args.initial_cash, now=now)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
