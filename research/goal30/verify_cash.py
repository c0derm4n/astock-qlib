"""复核已完成现金回放的冻结输入、事件日期、归档账本和收益指标，不重新选参。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, market30
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, SELECTED
from research.goal30.cash_replay import actions_from_evidence, audit_cash_ledger
from research.sp500.validate import relative_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = Path(directory).resolve()
    frozen = json.loads((directory / 'frozen_protocol.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    selected, _ = load_selected(SELECTED)
    for path, expected in frozen['input_hashes'].items():
        require(sha256(BASE / path) == expected, f'冻结输入不一致：{path}')
    require(selected['candidate'] == frozen['candidate'] and selected['policy'] == frozen['policy'],
            '回放规则不等于原冻结候选')
    require(len(SYMBOLS) == 30 and 'SH513100' not in SYMBOLS, '交易池不是指定30只')
    protocol = frozen['protocol']
    evidence = json.loads((BASE / protocol['actions_source']).read_text(encoding='utf-8'))
    require(actions_from_evidence(evidence, protocol['start'], protocol['end']) == frozen['actions'],
            '归档公司行动不等于已取得的证据')
    require(set(result['cases']) == {case['name'] for case in protocol['cases']}, '情景缺失或多出')
    market = market30(BASE / selected['source'])
    expected_dates = market.calendar[(market.calendar >= protocol['start']) &
                                     (market.calendar <= protocol['end'])]
    expected_dates = expected_dates.astype('datetime64[ns]')
    require(expected_dates[-1] == market.calendar[-1], '行情截止日不一致')
    checked = {}
    for case in protocol['cases']:
        name = case['name']
        reported = result['cases'][name]
        report = pd.read_csv(directory / f'report_{name}.csv', index_col=0, parse_dates=True)
        trades = pd.read_csv(directory / f'trades_{name}.csv')
        events = pd.read_csv(directory / f'actions_{name}.csv')
        holdings = pd.read_csv(directory / f'holdings_{name}.csv', parse_dates=['date'])
        require(report.index.equals(expected_dates), f'{name}日期缺失或重复')
        require(set(trades.symbol).issubset(SYMBOLS), f'{name}存在池外成交')
        require(not holdings.duplicated(['date', 'symbol']).any(), '持仓日期/标的重复')
        require((holdings.quantity > 0).all(), '持仓份额必须为正')
        quantities = holdings.set_index(['date', 'symbol']).quantity
        close = market.wide('close').ffill(limit=5)
        for row in holdings.itertuples(index=False):
            require(np.isclose(row.value, row.quantity * close.at[row.date, row.symbol], atol=.001, rtol=0),
                    '归档持仓市值不等于原价乘份额')
        values = holdings.groupby('date').value.sum().reindex(report.index, fill_value=0.)
        require(np.allclose(values, report.value, atol=.001, rtol=0), '归档持仓总值不等于账本市值')
        require(not events.duplicated(['event_id', 'type']).any(), '公司行动重复')
        require(set(events.event_id) == {event['event_id'] for event in frozen['actions']}, '公司行动缺漏')
        expected_rows = 0
        for event in frozen['actions']:
            group = events[events.event_id.eq(event['event_id'])]
            symbol = event['symbol']
            require(group.symbol.eq(symbol).all(), '公司行动标的不一致')
            if event['type'] == 'split':
                expected_rows += 1
                require(len(group) == 1, '拆分记录数量错误')
                row = group.iloc[0]
                date = pd.Timestamp(event['effective_date'])
                require(row.type == 'split' and row.phase == 'pre_open' and row.date == str(date.date()),
                        '拆分执行时间错误')
                require(row.ratio == event['ratio'], '拆分比例错误')
                previous = market.calendar[market.calendar.get_loc(date) - 1]
                before = quantities.get((previous, symbol), 0.)
                after = quantities.get((date, symbol), 0.)
                require(row.quantity_before == before and row.quantity_after == after and
                        np.isclose(after, before * event['ratio']), '拆分与归档持仓不一致')
            else:
                expected_rows += 3
                require(len(group) == 3, '分红必须有登记、除息和支付记录')
                quantity = quantities.get((pd.Timestamp(event['record_date']), symbol), 0.)
                amount = quantity * event['cash_per_share']
                payment = market.calendar.searchsorted(pd.Timestamp(event['payment_date']))
                payment += case['payment_delay_sessions']
                require(payment < len(market.calendar), '本次归档验收不支持区间外支付日')
                schedule = {
                    'record_entitlement': (event['record_date'], 'post_close'),
                    'accrue_dividend': (event['ex_date'], 'pre_open'),
                    'pay_dividend': (str(market.calendar[payment].date()), 'post_close'),
                }
                for row in group.itertuples(index=False):
                    require(row.type in schedule and (row.date, row.phase) == schedule[row.type],
                            '分红时点不等于冻结协议')
                    require(np.isclose(row.amount, amount, atol=.001, rtol=0), '分红不等于登记日份额权益')
                record = group[group.type.eq('record_entitlement')].iloc[0]
                require(record.quantity == quantity and
                        np.isclose(record.cash_per_share, event['cash_per_share']), '登记权益错误')
        require(len(events) == expected_rows, '公司行动记录数错误')
        ledger = audit_cash_ledger(market, report, trades, events, frozen['policy']['initial_capital'])
        for key in ('rows', 'trades', 'actions', 'violations'):
            require(ledger[key] == reported['ledger'][key], f'{name}账本摘要不一致：{key}')
        paid = float(events.loc[events.type.eq('pay_dividend'), 'amount'].sum())
        require(np.isclose(paid, reported['cash_dividends_received']), '分红到账汇总错误')
        require(np.isclose(report.dividend_receivable.max(), reported['max_dividend_receivable']),
                '应收分红峰值错误')
        net = report['return'] - report.cost
        prior = report.account.shift(1).fillna(frozen['policy']['initial_capital'])
        require(np.allclose(net, report.account / prior - 1, atol=1e-12, rtol=0), '每日收益与净值不一致')
        _, benchmark = relative_metrics(report, BASE / selected['benchmark'], selected['comparison'])
        require(benchmark.index.equals(report.index), '基准与策略交易日不一致')
        # 从归档逐日净值独立计算，不调用待核验的 comparison/wealth_metrics。
        metrics = {}
        for label, wealth in [('strategy', report.account), ('spy_total_cny', benchmark.spy_total_cny)]:
            dd = 1 - wealth / wealth.cummax()
            annual = {}
            for year in range(wealth.index[0].year + 1, wealth.index[-1].year):
                block = pd.concat([wealth[wealth.index.year < year].tail(1), wealth[wealth.index.year == year]])
                annual[str(year)] = float((1 - block / block.cummax()).max())
            terminal = float(wealth.iloc[-1] / wealth.iloc[0])
            metrics[label] = {'cumulative_return': terminal - 1,
                'cagr': terminal ** (365.25 / (wealth.index[-1] - wealth.index[0]).days) - 1,
                'mean_daily_drawdown': float(dd.mean()), 'mean_annual_max_drawdown': float(np.mean(list(annual.values()))),
                'max_drawdown': float(dd.max())}
            for key, value in metrics[label].items():
                require(np.isclose(value, reported[label][key], atol=1e-12, rtol=0), f'{name}/{label}/{key}不一致')
            for year, value in annual.items():
                require(np.isclose(value, reported[label]['annual_max_drawdowns'][year], atol=1e-12, rtol=0),
                        '逐年最大回撤不一致')
        a, b = metrics['strategy'], metrics['spy_total_cny']
        cagr = a['cagr'] >= b['cagr'] - 1e-12
        annual_ok = a['mean_annual_max_drawdown'] <= b['mean_annual_max_drawdown'] + 1e-12
        daily_ok = a['mean_daily_drawdown'] <= b['mean_daily_drawdown'] + 1e-12
        checks = {'cagr_at_least_benchmark': cagr, 'mean_annual_drawdown_no_worse': annual_ok,
                  'mean_daily_drawdown_no_worse': daily_ok, 'annual_definition_target_met': cagr and annual_ok,
                  'daily_definition_target_met': cagr and daily_ok,
                  'both_definitions_target_met': cagr and annual_ok and daily_ok}
        require(checks == reported['checks'], '收益/回撤判断不一致')
        checked[name] = {'ledger': ledger, 'metrics': metrics, 'checks': checks,
                         'cash_dividends_received': paid, 'event_schedule_verified': True}
    require(all(result[key] is False for key in ('corporate_actions_audited',
            'historical_universe_audited', 'investment_admission')), '不能将本次核验包装为全部准入')
    return {'verified_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'directory': str(directory), 'selected_strategy_frozen_inputs_unchanged': True,
        'frozen_input_hashes_verified': len(frozen['input_hashes']), 'universe_count': len(SYMBOLS),
        'cases': checked, 'overall_goal_complete': False,
        'evidence_hashes': {p.relative_to(BASE).as_posix(): sha256(p) for p in
                           [*directory.glob('*.csv'), directory / 'results.json', directory / 'frozen_protocol.json',
                            Path(__file__), BASE / 'research/goal30/audit.py']},
        'integrity_note': '结果文件指纹为本次复核时建立；公司行动正文及历史目录审计仍未完成。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verified = verify(args.directory)
    write_json_atomic(args.output, verified)
    print(json.dumps({'output': str(args.output), 'cases_verified': len(verified['cases']),
                      'frozen_input_hashes_verified': verified['frozen_input_hashes_verified']}, ensure_ascii=False))
