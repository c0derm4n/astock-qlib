"""从归档文件独立复核动态配置的指标、现金、时点及信号，不重新运行策略。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import SYMBOLS, market30
from research.goal30.relative_allocation import verify_trace

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / 'output/goal30/relative_allocation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_metrics(wealth):
    require(len(wealth) > 1 and wealth.index.is_unique and wealth.index.is_monotonic_increasing,
            '净值日期缺失、重复或乱序')
    require(np.isfinite(wealth).all() and wealth.gt(0).all(), '净值含非正或非有限数')
    drawdown = 1 - wealth / wealth.cummax()
    annual_dd = {}
    for year in range(wealth.index[0].year + 1, wealth.index[-1].year):
        before = wealth[wealth.index.year < year].tail(1)
        block = pd.concat([before, wealth[wealth.index.year == year]])
        require(len(block) > 1, '缺少完整中间年份')
        annual_dd[str(year)] = float((1 - block / block.cummax()).max())
    terminal = wealth.iloc[-1] / wealth.iloc[0]
    return {'cagr': float(terminal ** (365.25 / (wealth.index[-1] - wealth.index[0]).days) - 1),
            'cumulative_return': float(terminal - 1), 'mean_daily_drawdown': float(drawdown.mean()),
            'mean_annual_max_drawdown': float(np.mean(list(annual_dd.values()))),
            'max_drawdown': float(drawdown.max()), 'annual_max_drawdowns': annual_dd}


def verify(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT), '只能核验动态配置研究目录')
    completion = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
    require(digest(directory / 'results.json') == completion['results_sha256'], '完成后的结果文件已变动')
    require(digest(directory / 'registration.json') == completion['registration_sha256'], '登记文件已变动')
    registry = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))
    result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    for path, expected in registry['input_hashes'].items():
        require(digest(BASE / path) == expected, f'冻结输入改变：{path}')
    spec = registry['protocol']
    expected_cases = {v['name'] + '__' + s['name'] for v in spec['variants'] for s in spec['scenarios']}
    require(set(result['cases']) == expected_cases and len(expected_cases) == completion['cases'] == 9,
            '登记9个情景没有全部完成')
    source = BASE / registry['source']
    market = market30(source)
    eligibility = market.eligibility.set_index(['date', 'symbol']).eligible
    bars = pd.read_parquet(source / 'market_snapshot.parquet')
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars.date).unique()))
    dates = dates[(dates >= spec['start']) & (dates <= spec['end'])]
    close = bars.pivot(index='date', columns='symbol', values='close').ffill(limit=5)
    opened = bars.pivot(index='date', columns='symbol', values='open')
    reference = BASE / registry['benchmark']
    spy = pd.read_csv(reference / 'spy.csv', index_col='date', parse_dates=True).adjusted_close
    fx = pd.read_csv(reference / 'usdcny.csv', index_col='date', parse_dates=True).close
    spy = spy.reindex(dates, method='ffill', tolerance=pd.Timedelta(days=7))
    fx = fx.reindex(dates, method='ffill', tolerance=pd.Timedelta(days=7))
    canonical_benchmark = spy * fx
    canonical_benchmark /= canonical_benchmark.iloc[0]
    benchmark_metrics = independent_metrics(canonical_benchmark)
    reviewed = {}
    for name, item in result['cases'].items():
        variant = next(v for v in spec['variants'] if v['name'] == item['variant']['name'])
        scenario = next(s for s in spec['scenarios'] if s['name'] == item['scenario']['name'])
        require(item['variant'] == variant and item['scenario'] == scenario
                and name == variant['name'] + '__' + scenario['name'], '归档方法/情景与登记不符')
        for filename, expected in item['artifact_hashes'].items():
            require(digest(directory / filename) == expected, f'{name}归档文件已变动：{filename}')
        report = pd.read_csv(directory / f'report_{name}.csv', index_col=0, parse_dates=True)
        benchmark = pd.read_csv(directory / f'benchmark_{name}.csv', index_col=0, parse_dates=True)
        trades = pd.read_csv(directory / f'trades_{name}.csv', parse_dates=['date', 'signal_date'])
        events = pd.read_csv(directory / f'actions_{name}.csv', parse_dates=['date'])
        holdings = pd.read_csv(directory / f'holdings_{name}.csv', parse_dates=['date'])
        signals = pd.read_csv(directory / f'signals_{name}.csv', parse_dates=['date'])
        require(report.index.equals(dates) and benchmark.index.equals(dates), f'{name}交易日不完整')
        require(pd.DatetimeIndex(signals.date).equals(dates[:-1]), f'{name}信号日期不完整')
        signal_check = verify_trace(market, spec, variant, registry['policy'],
                                   signals.assign(date=signals.date.dt.strftime('%Y-%m-%d')).to_dict('records'))
        require(np.allclose(benchmark.spy_total_cny, canonical_benchmark, atol=1e-12, rtol=0), '归档基准不等于原SPY乘汇率')
        strategy_metrics = independent_metrics(report.account)
        for label, metrics in [('strategy', strategy_metrics), ('spy_total_cny', benchmark_metrics)]:
            for key, actual in metrics.items():
                if isinstance(actual, dict):
                    require(actual.keys() == item[label][key].keys(), '年度回撤覆盖年份不同')
                    require(all(abs(v - item[label][key][y]) < 1e-12 for y, v in actual.items()), '年度回撤不一致')
                else:
                    require(abs(actual - item[label][key]) < 1e-12, f'{name}/{label}/{key}独立复算不一致')
        annual = report.account.resample('YE').last()
        returns = annual.pct_change(fill_method=None)
        returns.iloc[0] = annual.iloc[0] / registry['policy']['initial_capital'] - 1
        require(all(abs(float(value) - item['annual_returns'][str(day.year)]) < 1e-12
                    for day, value in returns.items()), '逐年收益不一致')
        require(set(trades.side).issubset({'buy', 'sell'}), '成交方向错误')
        require(set(trades.symbol).issubset(spec['signal_symbols']) and set(holdings.symbol).issubset(spec['signal_symbols']),
                '成交或持仓出现范围之外标的')
        signal_indices = dates.get_indexer(trades.signal_date)
        execution_indices = dates.get_indexer(trades.date)
        require((signal_indices >= 0).all() and (execution_indices == signal_indices + 1).all(),
                '成交不是信号的下一交易日')
        require(np.isfinite(trades[['quantity', 'price', 'commission', 'slippage_cost', 'participation']]).all().all()
                and trades.quantity.gt(0).all() and trades.price.gt(0).all()
                and trades.commission.ge(0).all() and trades.slippage_cost.ge(0).all()
                and trades.participation.between(0, registry['policy']['max_adv_participation'] + 1e-10).all(),
                '成交数值、费用或参与率无效')
        for trade in trades[trades.side.eq('buy')].itertuples():
            require(trade.quantity % 100 == 0 and bool(eligibility.loc[(trade.signal_date, trade.symbol)]),
                    '买入不是整手或信号时不合格')
        require(not holdings.duplicated(['date', 'symbol']).any(), '持仓重复')
        require(holdings.quantity.gt(0).all(), '持仓含非正份额')
        values = holdings.quantity.to_numpy() * np.array([close.at[row.date, row.symbol] for row in holdings.itertuples()])
        require(np.allclose(values, holdings.value, atol=1e-6, rtol=0), '持仓估值与原价不一致')
        daily_value = holdings.groupby('date').value.sum().reindex(dates, fill_value=0.)
        signed = np.where(trades.side.eq('buy'), -1, 1)
        trades['cash_change'] = signed * trades.quantity * trades.price - trades.commission
        changes = trades.groupby('date').cash_change.sum().reindex(dates, fill_value=0.)
        payments = events[events.type.eq('pay_dividend')].groupby('date').amount.sum().reindex(dates, fill_value=0.)
        accrued = events[events.type.eq('accrue_dividend')].groupby('date').amount.sum().reindex(dates, fill_value=0.)
        expected_claims = (accrued - payments).cumsum()
        require(np.allclose(expected_claims, report.dividend_receivable, atol=1e-6, rtol=0), '应收权益与事件流水不一致')
        quantities = holdings.set_index(['date', 'symbol']).quantity
        quantity_frame = holdings.pivot(index='date', columns='symbol', values='quantity').reindex(dates).fillna(0.)
        previous_quantities = quantity_frame.shift(1, fill_value=0.)
        signed_quantity = trades.assign(delta=np.where(trades.side.eq('buy'), 1, -1) * trades.quantity)
        quantity_changes = signed_quantity.groupby(['date', 'symbol']).delta.sum().unstack().reindex_like(quantity_frame).fillna(0.)
        expected_quantities = previous_quantities + quantity_changes
        for split in events[events.type.eq('split')].itertuples():
            if split.symbol in expected_quantities:
                expected_quantities.at[split.date, split.symbol] += previous_quantities.at[split.date, split.symbol] * (split.ratio - 1)
        require(np.allclose(expected_quantities, quantity_frame, rtol=0, atol=1e-8), '逐日持仓不能由成交和拆分重建')
        require(set(events.event_id) == {e['event_id'] for e in registry['actions']}, '公司行动缺失或多出')
        for event in registry['actions']:
            group = events[events.event_id.eq(event['event_id'])]
            require(group.symbol.eq(event['symbol']).all(), '公司行动标的不一致')
            if event['type'] == 'cash_dividend':
                quantity = quantities.get((pd.Timestamp(event['record_date']), event['symbol']), 0.)
                amount = quantity * event['cash_per_share']
                require(len(group) == 3 and np.allclose(group.amount, amount, atol=1e-6, rtol=0),
                        '分红与登记日真实份额不一致')
                payment = dates.searchsorted(pd.Timestamp(event['payment_date'])) + item['scenario']['payment_delay_sessions']
                schedule = {'record_entitlement': (pd.Timestamp(event['record_date']), 'post_close'),
                            'accrue_dividend': (pd.Timestamp(event['ex_date']), 'pre_open'),
                            'pay_dividend': (dates[payment], 'post_close')}
                require(set(group.type) == set(schedule), '分红阶段缺失或重复')
                require(all((r.date, r.phase) == schedule[r.type] for r in group.itertuples()), '分红事件时点错误')
            else:
                require(len(group) == 1, '拆分重复')
                row = group.iloc[0]
                day = pd.Timestamp(event['effective_date'])
                previous = dates[dates.get_loc(day) - 1]
                before = quantities.get((previous, event['symbol']), 0.)
                after = quantities.get((day, event['symbol']), 0.)
                require(row.date == day and row.type == 'split' and row.phase == 'pre_open'
                        and row.ratio == event['ratio'] and row.quantity_before == before
                        and row.quantity_after == after and abs(after - before * event['ratio']) < 1e-8,
                        '拆分日期、比例或持仓份额不一致')
        expected_cash = registry['policy']['initial_capital'] + (changes + payments).cumsum()
        cash_error = float((expected_cash - report.cash).abs().max())
        value_error = float((daily_value - report.value).abs().max())
        nav_error = float((expected_cash + daily_value + report.dividend_receivable - report.account).abs().max())
        require(max(cash_error, value_error, nav_error) < 1e-6, '现金流/归档持仓/经济净值不一致')
        prior = report.account.shift(1).fillna(registry['policy']['initial_capital'])
        traded_at_open = trades.assign(notional=trades.quantity.to_numpy() *
                            np.array([opened.at[row.date, row.symbol] for row in trades.itertuples()]))
        turnover = traded_at_open.groupby('date').notional.sum().reindex(dates, fill_value=0.) / prior
        require(np.allclose(turnover, report.turnover, atol=1e-12, rtol=0), '逐日双边换手与成交额不一致')
        turnover_limit = registry['policy']['max_daily_turnover']
        execution_constraints = {'turnover_parameter': turnover_limit, 'maximum_actual_daily_turnover': float(turnover.max()),
            'sessions_above_turnover_parameter': int((turnover > turnover_limit + 1e-10).sum()),
            'hard_daily_turnover_cap_satisfied': bool((turnover <= turnover_limit + 1e-10).all()),
            'semantics': '原执行器以该参数约束买入预算，清仓卖出与次日价格变化可使实际双边换手超过参数；不声称满足硬上限。'}
        require(np.allclose(report['return'] - report.cost, report.account / prior - 1, atol=1e-12, rtol=0),
                '每日扣费收益与经济净值不一致')
        a, b = strategy_metrics, benchmark_metrics
        cagr = a['cagr'] >= b['cagr'] - 1e-12
        annual_ok = a['mean_annual_max_drawdown'] <= b['mean_annual_max_drawdown'] + 1e-12
        daily_ok = a['mean_daily_drawdown'] <= b['mean_daily_drawdown'] + 1e-12
        max_ok = a['max_drawdown'] <= registry['policy']['max_drawdown_acceptance']
        checks = {'cagr_at_least_benchmark': cagr, 'mean_annual_drawdown_no_worse': annual_ok,
                  'mean_daily_drawdown_no_worse': daily_ok, 'annual_definition_target_met': cagr and annual_ok,
                  'daily_definition_target_met': cagr and daily_ok, 'both_definitions_target_met': cagr and annual_ok and daily_ok,
                  'max_drawdown_within_original_limit': max_ok, 'annual_goal_and_max_drawdown': cagr and annual_ok and max_ok,
                  'daily_goal_and_max_drawdown': cagr and daily_ok and max_ok}
        require(checks == item['checks'], '通过/失败判定不一致')
        reviewed[name] = {'checks': checks, 'cash_error': cash_error, 'value_error': value_error, 'nav_error': nav_error,
                          'signal_check': signal_check, 'daily_holdings_rebuilt': True, 'next_session_execution_verified': True,
                          'execution_constraints': execution_constraints,
                          'metrics': a, 'artifacts_verified': len(item['artifact_hashes'])}
    verification = {'verified_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                    'cases_verified': len(reviewed), 'cases': reviewed, 'benchmark': benchmark_metrics,
                    'input_hashes_verified': len(registry['input_hashes']),
                    'verification_code_sha256': digest(Path(__file__)),
                    'results_sha256': digest(directory / 'results.json'), 'full_goal_achieved': False}
    write_json_atomic(directory / 'verification.json', verification)
    print(json.dumps({'cases_verified': len(reviewed), 'input_hashes_verified': len(registry['input_hashes']),
                      'annual_goal_and_max_drawdown_passes': sum(r['checks']['annual_goal_and_max_drawdown'] for r in reviewed.values()),
                      'daily_goal_and_max_drawdown_passes': sum(r['checks']['daily_goal_and_max_drawdown'] for r in reviewed.values())},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
