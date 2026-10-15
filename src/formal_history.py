"""按真实回放持仓生成最近十个自然周的收盘快照。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import exchange_calendars as xcals
import universe


def weekly_holdings_history(report, holdings, weeks=10):
    if report.empty:
        return []
    report = report.copy().sort_index()
    report.index = pd.DatetimeIndex(report.index).as_unit('ns')
    positions = holdings.copy()
    if not positions.empty:
        positions['date'] = pd.to_datetime(positions['date']).astype('datetime64[ns]')
        if positions.duplicated(['date', 'symbol']).any():
            raise ValueError('历史持仓存在重复日期/标的')
    grouped = {date: part for date, part in positions.groupby('date')} if not positions.empty else {}
    last_week = report.index[-1].to_period('W-SUN')
    first_week = max(report.index[0].to_period('W-SUN'), last_week - (weeks - 1))
    calendar = xcals.get_calendar('XSHG')
    result = []
    for week in reversed(pd.period_range(first_week, last_week, freq='W-SUN')):
        start, end = week.start_time.normalize(), week.end_time.normalize()
        observed = report[(report.index >= start) & (report.index <= end)]
        sessions = calendar.sessions_in_range(start, end)
        carried = observed.empty
        if carried and len(sessions):
            raise ValueError(f'{start.date()}所在周有交易日却没有回放记录，不能虚构历史持仓')
        available = report.loc[:end]
        if available.empty:
            continue
        date, row = available.index[-1], available.iloc[-1]
        nav, cash = float(row.account), float(row.cash)
        claims = float(row.get('dividend_receivable', 0))
        items = []
        for p in grouped.get(date, pd.DataFrame()).to_dict('records'):
            q, value = float(p['quantity']), float(p['value'])
            if q <= 1e-8:
                continue
            symbol = p['symbol']
            items.append({'symbol': symbol, 'code': symbol[2:], 'name': universe.UNIVERSE.get(symbol[2:], symbol),
                          'quantity': q, 'value': value, 'weight': value / nav})
        stock_value = sum(p['value'] for p in items)
        if not np.isclose(stock_value, float(row.value), rtol=1e-10, atol=.01):
            raise ValueError(f'{date.date()}历史持仓市值与账户报告不一致')
        if not np.isclose(stock_value + cash + claims, nav, rtol=1e-10, atol=.01):
            raise ValueError(f'{date.date()}历史持仓、现金和分红应收未对平')
        result.append({'week_start': str(start.date()), 'week_end': str(end.date()),
                       'date': str(date.date()), 'partial': bool(len(sessions) and sessions[-1] > report.index[-1]),
                       'carried_over': carried, 'holdings': sorted(items, key=lambda p: (-p['weight'], p['symbol'])),
                       'account': nav, 'cash': cash, 'cash_weight': cash / nav,
                       'dividend_receivable': claims, 'receivable_weight': claims / nav,
                       'exposure': stock_value / nav, 'empty': not bool(items)})
    return result
