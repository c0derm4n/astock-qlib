"""双策略正式服务产物构建；独立数据库、不可变运行目录、原子当前指针。"""
from __future__ import annotations
import argparse
import copy
import json
import pickle
import shutil
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from filelock import FileLock
import universe
from src.artifacts import write_json_atomic, json_safe
from src.runs import sha256
from src.market import load_market
from src.portfolio import target_weights
from src.formal_orders import plan_orders
from src.formal_rules import ROOT, SYMBOLS, registry, weekly_sessions, annual_lightgbm, trend_targets
from src.formal_adjusted import simulate
from src.formal_cash import simulate_cash, audit_cash_ledger
from src.formal_history import weekly_holdings_history

OUT = ROOT / 'output/strategies'
DATABASE = ROOT / 'data/formal/market.duckdb'
SEED = ROOT / 'output/history_backfill_2016/side_20261013/market_2013_2026.duckdb'


def now():
    return pd.Timestamp.now(tz='Asia/Shanghai')


def sessions(start, end):
    return xcals.get_calendar('XSHG').sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def completed_day(at=None):
    at = at if at is not None else now()
    day = at.tz_localize(None).normalize()
    dates = sessions(day - pd.Timedelta(days=20), day)
    if at.hour < 19:
        dates = dates[dates < day]
    return dates[-1]


def next_session(day):
    return sessions(pd.Timestamp(day) + pd.Timedelta(days=1), pd.Timestamp(day) + pd.Timedelta(days=30))[0]


def runtime_status(payload, at=None):
    """随请求时间重算时效；旧快照不能永久保持可执行。"""
    at = at if at is not None else now()
    expected = completed_day(at)
    stale = pd.Timestamp(payload['data_end']) < expected
    signal = pd.Timestamp(payload['advice']['signal_date'])
    execute = next_session(signal)
    today = at.tz_localize(None).normalize()
    if stale:
        status, label = 'stale', '行情过期 · 请先刷新'
    elif execute < today or (execute == today and (at.hour, at.minute) >= (9, 30)):
        status, label = 'elapsed', '本周开盘执行窗口已过'
    else:
        status, label = 'pending', '下一交易日开盘待评估'
    future = sessions(today - pd.Timedelta(days=14), today + pd.Timedelta(days=30))
    first = weekly_sessions(future, future[0])
    remaining = first[(first > today) | ((first == today) & (at.hour < 19))]
    return {'expected_data_end': str(expected.date()), 'stale': stale, 'advice_status': status,
            'advice_label': label, 'execution_date': str(execute.date()),
            'next_signal_date': str(remaining[0].date()),
            'can_execute': False, 'paper_only': True}


def stats(report, capital):
    nav = report.account
    dd = nav / nav.cummax().clip(lower=capital) - 1
    annual, prior = [], capital
    for year, values in nav.groupby(nav.index.year):
        complete = year > nav.index[0].year and year < nav.index[-1].year
        # 当年数据已覆盖该年最后一个交易日，末年才视为完整。
        if year == nav.index[-1].year and values.index[-1] == sessions(f'{year}-12-01', f'{year}-12-31')[-1]:
            complete = year > nav.index[0].year
        curve = pd.concat([pd.Series([prior]), pd.Series(values.to_numpy())], ignore_index=True)
        annual.append({'year': int(year), 'return': float(values.iloc[-1] / prior - 1),
                       'max_drawdown': float((1 - curve / curve.cummax()).max()), 'complete': bool(complete)})
        prior = float(values.iloc[-1])
    worst = min((x for x in annual if x['complete']), key=lambda x: x['return'], default=None)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {'final_account': float(nav.iloc[-1]), 'cumulative_return': float(nav.iloc[-1] / capital - 1),
            'cagr': float((nav.iloc[-1] / capital) ** (1 / years) - 1),
            'max_drawdown': float(-dd.min()), 'worst_complete_year': worst,
            'annual': annual, 'observations': len(nav)}, dd


def holdings_rows(quantities, cash, prices):
    nav = cash + sum(q * prices[s] for s, q in quantities.items())
    return [{'symbol': s, 'code': s[2:], 'name': universe.UNIVERSE.get(s[2:], s),
             'quantity': float(q), 'price': float(prices[s]), 'value': float(q * prices[s]),
             'weight': float(q * prices[s] / nav)} for s, q in sorted(quantities.items()) if q > 1e-8]


def advice_rows(targets, orders, quantities, cash, prices, eligible, strategy_id):
    nav = cash + sum(q * prices[s] for s, q in quantities.items())
    order_map = {o['symbol']: o for o in orders}
    rows = []
    for s in sorted(set(targets) | set(quantities)):
        o = order_map.get(s)
        if quantities.get(s, 0) < 1 and targets.get(s, 0) == 0 and (not o or o['quantity'] < 1):
            continue
        current = quantities.get(s, 0) * prices[s] / nav
        side = o['side'] if o else 'hold'
        if s not in eligible:
            reason = '市场资格失效，逐步退出'
        elif targets.get(s, 0) == 0 and strategy_id == 'trend200_vol15':
            reason = '趋势门槛不满足或尚在退出'
        elif o:
            reason = '目标偏离触发；已应用风险、现金与成交额约束'
        else:
            reason = '保持；漂移阈值、最小持有期或整数份额约束'
        q = float(o['quantity']) if o else 0.
        rows.append({'symbol': s, 'code': s[2:], 'name': universe.UNIVERSE.get(s[2:], s),
                     'side': side, 'quantity': q, 'reference_price': float(prices[s]),
                     'estimated_amount': float(q * prices[s]), 'current_weight': float(current),
                     'target_weight': float(targets.get(s, 0)), 'reason': reason})
    return rows


def preview(context, definition, cash, holdings):
    """真实账户试算使用与回放一致的目标与订单规则，输入不落盘。"""
    quantities = {h['symbol']: h['quantity'] for h in holdings}
    days = {h['symbol']: h['holding_days'] for h in holdings}
    if set(quantities) - set(context['prices']):
        raise ValueError('持仓包含当前策略池外或无价格标的')
    prices = context['prices']
    nav = cash + sum(q * prices[s] for s, q in quantities.items())
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError('账户总资产必须大于0')
    current = {s: q * prices[s] / nav for s, q in quantities.items()}
    returns = pd.DataFrame(context['returns'])
    returns.index = pd.to_datetime(returns.index)
    policy = definition['policy']
    if definition['id'] == 'lightgbm':
        targets = target_weights(pd.Series(context['scores'], dtype=float), context['eligible'], current, days,
                                 returns, policy) if context.get('model_available', True) else {}
    else:
        history = pd.DataFrame(context['trend_prices'])
        history.index = pd.to_datetime(history.index)
        targets = trend_targets(definition['candidate'], pd.Timestamp(context['date']),
                                context['eligible'], current, returns, history, policy)
    orders = plan_orders(targets, quantities, cash, prices, context['adv'], policy)
    return {'account': nav, 'cash': cash, 'signal_date': context['date'], 'targets': targets,
            'rows': advice_rows(targets, orders, quantities, cash, prices, context['eligible'], definition['id']),
            'paper_only': True, 'basis': '输入账户在上述历史信号收盘时点的持仓与可用现金；不预支卖出所得'}


def context_at(market, date, scores, allowed_symbols):
    allowed = market.eligibility[market.eligibility.date.eq(date)].set_index('symbol')
    prices = market.wide('close').ffill(limit=5).loc[date].reindex(allowed_symbols).dropna().to_dict()
    returns = market.wide('adj_close').pct_change(fill_method=None).loc[:date].tail(60)
    returns = returns.reindex(columns=allowed_symbols)
    returns.index = returns.index.strftime('%Y-%m-%d')
    history = market.wide('adj_close').loc[:date].tail(200).reindex(columns=['SH515880', 'SH518880', 'SH510300'])
    history.index = history.index.strftime('%Y-%m-%d')
    return json_safe({'date': str(date.date()), 'prices': prices, 'scores': scores.to_dict(),
                      'eligible': sorted(set(allowed.index[allowed.eligible]) & set(allowed_symbols)),
                      'adv': allowed.avg_amount.reindex(allowed_symbols).fillna(0).to_dict(),
                      'returns': returns.to_dict(), 'trend_prices': history.to_dict()})


def verify_execution(market, report, trades, start):
    wanted = market.calendar[market.calendar >= pd.Timestamp(start)]
    if list(report.index.strftime('%Y-%m-%d')) != list(wanted.strftime('%Y-%m-%d')):
        raise ValueError('资金曲线必须覆盖共同起点后的全部真实交易日')
    if not report.cash.ge(-1e-6).all():
        raise ValueError('回放出现负现金')
    if not trades.empty:
        signals = pd.to_datetime(trades.signal_date)
        indices = market.calendar.searchsorted(signals)
        if not signals.isin(weekly_sessions(market.calendar, start)).all() or not np.array_equal(
                market.calendar[indices + 1].strftime('%Y-%m-%d'), pd.to_datetime(trades.date).dt.strftime('%Y-%m-%d')):
            raise ValueError('存在非周频或非T+1交易')
        if not trades.participation.le(.020000001).all():
            raise ValueError('成交额参与率超限')


def check_new_actions(market, action_spec):
    """禁止把未取得支付日的新增分红静默忽略。"""
    for symbol in ['SH515880', 'SH518880', 'SH510300']:
        factors = market.wide('adj_factor')[symbol].dropna()
        changes = factors.pct_change().abs()
        unknown = changes[(changes > 1e-6) & (changes.index > pd.Timestamp(action_spec['verified_through']))]
        known = {pd.Timestamp(e.get('ex_date', e.get('effective_date'))) for e in action_spec['actions'] if e['symbol'] == symbol}
        if set(unknown.index) - known:
            raise ValueError(f'{symbol} 新增公司行动缺少登记/支付日证据，请补齐 strategy_corporate_actions.json')


def build(fetch=False):
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / 'refresh.lock'), timeout=0):
        if not DATABASE.exists():
            if not SEED.exists():
                raise ValueError('缺少2014/15年已补齐的历史基线，请恢复补数数据库')
            spec = json.loads(SEED.with_name('backfill_completion.json').read_text(encoding='utf-8'))
            if sha256(SEED) != spec['database_sha256']:
                raise ValueError('历史基线哈希不匹配')
            DATABASE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SEED, DATABASE)
        refresh_error = None
        if fetch:
            try:
                from src.formal_ingest import run_free_update
                run_free_update(DATABASE, OUT, end=completed_day())
            except Exception as exc:
                refresh_error = f'{type(exc).__name__}: {exc}'
                print(f'增量更新未通过，使用最后有效数据库：{refresh_error}', flush=True)
        market = load_market(DATABASE)
        spec = registry()
        action_spec = json.loads((ROOT / 'strategy_corporate_actions.json').read_text(encoding='utf-8'))
        check_new_actions(market, action_spec)
        run_id = now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        directory = OUT / 'runs' / run_id
        directory.mkdir(parents=True)
        source_files = [ROOT / 'strategy_registry.json', ROOT / 'strategy_corporate_actions.json',
                        *sorted((ROOT / 'src').glob('formal_*.py')), ROOT / 'src/portfolio.py',
                        ROOT / 'src/models.py', ROOT / 'src/market.py', ROOT / 'src/universe_filter.py']
        hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files}
        write_json_atomic(directory / 'registration.json', {'registered_at': now().isoformat(), 'registry': spec,
            'market_fingerprint': market.fingerprint, 'database_sha256': sha256(DATABASE),
            'input_hashes': hashes, 'data_audit': market.audit, 'refresh_error': refresh_error,
            'historical_selection_status': '已查看历史后的正式服务登记；非策略选择独立盲测'})
        predictions, folds, bundle = annual_lightgbm(market, spec['strategies'][0]['policy'])
        predictions.to_pickle(directory / 'predictions_lightgbm.pkl')
        with (directory / 'annual_model.pkl').open('wb') as stream:
            pickle.dump(bundle, stream)
        write_json_atomic(directory / 'folds.json', {'folds': folds})
        strategies = []
        signal_days = weekly_sessions(market.calendar, spec['start'])
        for definition in spec['strategies']:
            id_ = definition['id']
            latest, final_state = {}, {}
            def observe(date, quantities, cash, bought, exiting, targets, orders):
                latest.clear()
                latest.update(date=date, quantities=quantities, cash=cash, bought=bought,
                              exiting=sorted(exiting), targets=targets, orders=copy.deepcopy(orders))
            if id_ == 'lightgbm':
                holdings_log = []
                mask = predictions.index.get_level_values('datetime').isin(signal_days)
                scores = predictions[mask]
                # 空预测周保留现状；共同起点缺预测则明确现金前缀。
                unavailable = signal_days[~signal_days.isin(scores.index.get_level_values('datetime').unique())]
                if len(unavailable):
                    dummy = pd.Series(np.nan, index=pd.MultiIndex.from_product(
                        [unavailable, ['SH510300']], names=['datetime', 'instrument']))
                    scores = pd.concat([dummy, scores])
                report, trades, issues = simulate(market, scores, definition['policy'], observer=observe,
                    state=final_state, unavailable_dates=set(unavailable), holdings_log=holdings_log)
                holdings = pd.DataFrame(holdings_log, columns=['date', 'symbol', 'quantity', 'value'])
                holdings.to_csv(directory / f'{id_}_holdings.csv', index=False)
                score = predictions.xs(latest['date'], level='datetime') if latest['date'] not in unavailable else pd.Series(dtype=float)
                allowed_symbols = sorted(SYMBOLS)
                ledger = None
            else:
                prices, returns = market.wide('adj_close'), market.wide('adj_close').pct_change(fill_method=None)
                def select(scores, eligible, current, days, history, policy, *, exiting=()):
                    return trend_targets(definition['candidate'], history.index[-1], eligible, current, history,
                                         prices, policy, exiting)
                report, trades, events, holdings, issues = simulate_cash(market, definition['candidate'],
                    definition['policy'], spec['start'], action_spec['actions'], selector=select,
                    observer=observe, state=final_state)
                ledger = audit_cash_ledger(market, report, trades, events, spec['initial_capital'])
                events.to_csv(directory / f'{id_}_actions.csv', index=False)
                holdings.to_csv(directory / f'{id_}_holdings.csv', index=False)
                score = pd.Series(dtype=float)
                allowed_symbols = ['SH515880', 'SH518880', 'SH510300']
            verify_execution(market, report, trades, spec['start'])
            context = context_at(market, latest['date'], score, allowed_symbols)
            context.update(exiting=latest['exiting'])
            context['model_available'] = bool(len(score)) if id_ == 'lightgbm' else True
            write_json_atomic(directory / f'{id_}_context.json', context)
            metrics, drawdown = stats(report, spec['initial_capital'])
            checks = {'cagr': metrics['cagr'] >= spec['goals']['cagr'],
                      'worst_complete_year': bool(metrics['worst_complete_year'] and metrics['worst_complete_year']['return'] >= spec['goals']['worst_complete_year']),
                      'max_drawdown': metrics['max_drawdown'] <= spec['goals']['max_drawdown']}
            prices_last = market.wide('close').ffill(limit=5).iloc[-1].to_dict()
            weekly_history = weekly_holdings_history(report, holdings, weeks=50 if id_ == 'trend200_vol15' else 10)
            payload = {'id': id_, 'name': definition['name'], 'color': definition['color'],
                'status': 'formal_service', 'description': definition['description'], 'version': spec['version'],
                'account_basis': definition['account_basis'], 'initial_capital': spec['initial_capital'],
                'data_start': spec['start'], 'data_end': str(market.calendar[-1].date()), 'run_id': run_id,
                'metrics': metrics, 'goals': checks, 'investment_admission': False,
                'limitations': ['尚未达到全部收益/年度亏损目标', '存续标的与完整公司行动审计、真实前瞻验证尚未通过'],
                'holdings': holdings_rows(final_state['quantities'], final_state['cash'], prices_last),
                'cash': float(final_state['cash']), 'dividend_receivable': float(final_state.get('dividend_receivable', 0)),
                'advice': {'signal_date': str(latest['date'].date()), 'account': float(latest['cash'] + sum(
                    q * context['prices'][s] for s, q in latest['quantities'].items())), 'cash': float(latest['cash']),
                    'execution_date': str(next_session(latest['date']).date()),
                    'rows': advice_rows(latest['targets'], latest['orders'], latest['quantities'], latest['cash'],
                                        context['prices'], context['eligible'], id_),
                    'basis': '策略模型账户在信号日收盘的持仓；参考收盘价估算，开盘需再次检查价格、现金和资格',
                    'orders_count': len(latest['orders'])},
                'series': {'dates': report.index.strftime('%Y-%m-%d').tolist(), 'capital': report.account.tolist(),
                           'drawdown': (-drawdown).tolist(), 'cash': report.cash.tolist()},
                'weekly_holdings': weekly_history[:10],
                'validation': {'weekly_t_plus_1': True, 'ledger': ledger, 'trade_count': len(trades), 'issues': issues}}
            if id_ == 'trend200_vol15':
                payload['weekly_allocation'] = weekly_history
            payload['execution_revision'] = 'v3.0.1：保持目标不产生份额取整碎单；调仓建议采用整数委托'
            write_json_atomic(directory / f'{id_}.json', payload)
            report.to_csv(directory / f'{id_}_report.csv')
            trades.to_csv(directory / f'{id_}_trades.csv', index=False)
            strategies.append(payload)
            print(f'{definition["name"]} 完成：资金 {metrics["final_account"]:.2f}', flush=True)
        for p, expected in hashes.items():
            if sha256(ROOT / p) != expected:
                raise ValueError(f'运行中正式实现发生变化：{p}')
        write_json_atomic(directory / 'completion.json', {'run_id': run_id, 'completed_at': now().isoformat(),
            'artifact_hashes': {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
            'refresh_error': refresh_error})
        write_json_atomic(OUT / 'current.json', {'run_id': run_id, 'directory': directory.relative_to(ROOT).as_posix(),
            'completed_at': now().isoformat(), 'data_end': str(market.calendar[-1].date()), 'refresh_error': refresh_error})
        print(f'正式双策略快照已发布：{directory}', flush=True)
        return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='先尝试增量更新行情；失败保留最后有效数据')
    args = parser.parse_args()
    build(fetch=args.fetch)
