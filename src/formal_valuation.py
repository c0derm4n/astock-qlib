"""等权估值策略的正式数据快照、回放及服务字段适配。

计算模块在 src/valuation 独立维护；不导入研究目录，不修改冻结历史。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import universe

from src.artifacts import json_safe, write_json_atomic
from src.runs import sha256
from src.formal_history import weekly_holdings_history
from src.valuation.data import load_snapshot, build_calendar
from src.valuation.engine import replay
from src.valuation.verify import _recompute_case

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'valuation_strategy_protocol.json'
SEED = ROOT / 'data/formal/valuation/seed/manifest.json'
IDS = ('csi300_pe_equal', 'chinext_pb_equal')
FIELDS = {'000300': ('equalWeightedPeTtm', 'PE_TTM_EQUAL'),
          '399006': ('equalWeightedPbTtm', 'PB_ETFRUN_EQUAL')}
URLS = {'000300': 'https://www.etf.run/index/SSE/000300',
        '399006': 'https://www.etf.run/index/SZSE/399006'}
PREVIEW_REASON = '估值分批策略需要批次及冷却历史，暂不支持只凭现金/份额试算'


def extract_daily(html):
    def visit(obj):
        if isinstance(obj, dict):
            if 'indexDailyItems' in obj:
                return obj['indexDailyItems']
            children = obj.values()
        elif isinstance(obj, list):
            children = obj
        else:
            return None
        for child in children:
            found = visit(child)
            if found is not None:
                return found
        return None
    for match in re.finditer(r'self\.__next_f\.push\((\[.*?\])\)</script>', html, re.S):
        packet = json.loads(match.group(1))
        if len(packet) < 2 or not isinstance(packet[1], str) or 'indexDailyItems' not in packet[1]:
            continue
        block = packet[1]
        found = visit(json.loads(block[block.index(':') + 1:]))
        if found is not None:
            frame = pd.DataFrame(found)
            frame['date'] = pd.to_datetime(frame['date'])
            if frame.date.duplicated().any():
                raise ValueError('等权历史出现重复日期')
            return frame.sort_values('date')
    raise ValueError('免费估值页面未提供公开日频历史')


def extend_index_frames(frames, code, incoming, calendar, digest, fetched_at):
    """验证全历史重叠后只追加；历史修订不能静默进入已有账户。"""
    field, metric = FIELDS[code]
    incoming = incoming[incoming.date >= pd.Timestamp('2011-01-01')].copy()
    expected = calendar[(calendar >= pd.Timestamp('2011-01-01')) & (calendar <= incoming.date.max())]
    if not pd.DatetimeIndex(incoming.date).equals(expected):
        raise ValueError(f'{code} 日频估值缺少交易日')
    values = incoming[['close', field]].astype(float)
    if not np.isfinite(values).all().all() or (values <= 0).any().any():
        raise ValueError(f'{code} 估值或价格非正/非有限')
    old = frames['index_valuation']
    old = old[(old.index_id == code) & (old.metric == metric)].set_index('trade_date')['value']
    price = frames['index_price']
    price = price[price.index_id == code].set_index('trade_date')['close']
    incoming = incoming.set_index('date')
    overlap = old.index.intersection(incoming.index)
    if len(overlap) != len(old) or not np.allclose(old.loc[overlap], incoming.loc[overlap, field], rtol=1e-8, atol=1e-8) \
            or not np.allclose(price.loc[overlap], incoming.loc[overlap, 'close'], rtol=1e-8, atol=1e-8):
        raise ValueError(f'{code} 供应商历史发生修订或覆盖缩短，保留最后有效估值')
    extra = incoming[incoming.index > old.index.max()]
    if extra.empty:
        return
    # 继续采用已审查版本的下一交易日19:00可得假设，不改为当日可得。
    available = [calendar[calendar.get_loc(day) + 1] + pd.Timedelta(hours=19) for day in extra.index]
    prices = pd.DataFrame({'index_id': code, 'trade_date': extra.index, 'close': extra.close.to_numpy(),
                           'price_type': 'price_index_close', 'source_id': 'etfrun_public_html',
                           'effective_available_at': available, 'availability_basis': 'assumed_next_session_19',
                           'raw_sha256': digest})
    valuations = pd.DataFrame({'index_id': code, 'trade_date': extra.index, 'metric': metric,
        'value': extra[field].to_numpy(), 'aggregation_method': 'provider_declared_equal_weight',
        'financial_basis': 'TTM' if code == '000300' else 'provider_equalWeightedPbTtm_MRQ_unverified',
        'source_id': 'etfrun_public_html', 'source_published_at': pd.NaT,
        'effective_available_at': available, 'availability_basis': 'assumed_next_session_19',
        'fetched_at': fetched_at, 'revision_id': 'formal_append_only', 'raw_sha256': digest})
    frames['index_price'] = pd.concat([frames['index_price'], prices], ignore_index=True)
    frames['index_valuation'] = pd.concat([frames['index_valuation'], valuations], ignore_index=True)


def prepare_data(directory, market, action_spec, fetch=False, previous_manifest=None):
    """每次发布冻结独立输入；增量失败时保留有效估值并披露错误。"""
    parent_path = Path(previous_manifest) if previous_manifest else SEED
    parent = load_snapshot(parent_path)
    if parent.dataset_kind != 'real':
        raise ValueError('正式估值输入必须为真实数据')
    frames = {name: frame.copy() for name, frame in parent.frames.items()}
    from src.formal_rules import registry
    definitions = [d for d in registry()['strategies'] if d.get('kind') == 'valuation_bands']
    codes = {'000300' if d['id'] == IDS[0] else '399006' for d in definitions}
    symbols = {d['symbol'] for d in definitions}
    for name in ('index_price', 'index_valuation'):
        frames[name] = frames[name][frames[name].index_id.isin(codes)].copy()
    for name in ('etf_bars', 'etf_actions'):
        frames[name] = frames[name][frames[name].symbol.isin(symbols)].copy()
    destination = directory / 'valuation_data'
    destination.mkdir(exist_ok=False)
    end = market.calendar[-1]
    history_end = frames['index_valuation'].trade_date.max()
    calendar = build_calendar('2010-01-01', max(end, history_end) + pd.Timedelta(days=30))
    errors, sources = [], []
    if fetch:
        for code, url in URLS.items():
            if code not in codes:
                continue
            try:
                response = requests.get(url, timeout=(10, 30), headers={'User-Agent': 'astock-qlib/3.3.0'})
                response.raise_for_status()
                raw = destination / f'{code}.html'
                raw.write_bytes(response.content)
                incoming = extract_daily(response.content.decode('utf-8'))
                incoming = incoming[incoming.date <= end]
                # 种子可能含比正式行情更晚的历史，验证时保留这部分重叠。
                seed_end = frames['index_valuation'].query('index_id == @code').trade_date.max()
                if seed_end > end:
                    incoming = extract_daily(response.content.decode('utf-8'))
                    incoming = incoming[incoming.date <= max(end, seed_end)]
                extend_index_frames(frames, code, incoming, calendar, sha256(raw), pd.Timestamp.now(tz='Asia/Shanghai').isoformat())
                sources.append({'url': url, 'raw_file': raw.name, 'raw_sha256': sha256(raw), 'index_id': code})
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                errors.append(f'{code}: {type(exc).__name__}: {exc}')
    # 保留种子的冻结原价，只从正式数据库追加更新日期。
    for symbol in sorted(symbols):
        old = frames['etf_bars']
        last = old.loc[old.symbol == symbol, 'trade_date'].max()
        extra = market.bars[(market.bars.symbol == symbol) & (market.bars.date > last)].copy()
        if extra.empty:
            continue
        bars = pd.DataFrame({'symbol': symbol, 'trade_date': extra.date, 'open': extra.open,
            'high': extra.high, 'low': extra.low, 'close': extra.close, 'volume_shares': extra.volume * 100,
            'amount_cny': extra.amount, 'source_id': extra.source,
            'available_at': extra.date + pd.Timedelta(hours=16), 'raw_sha256': market.fingerprint})
        frames['etf_bars'] = pd.concat([old, bars], ignore_index=True)
        # 新公司行动必须有明确登记/支付日；历史事件保留种子证据口径。
        for action in action_spec['actions']:
            if action['symbol'] != symbol:
                continue
            ex = pd.Timestamp(action.get('ex_date', action.get('effective_date')))
            if ex <= last:
                continue
            if action['type'] != 'cash_dividend' or not action.get('record_date') or not action.get('payment_date'):
                raise ValueError(f'{symbol} 新公司行动缺合格登记/支付日证据')
            event = {'event_id': action['event_id'], 'symbol': symbol, 'type': 'dividend',
                'record_date': pd.Timestamp(action['record_date']), 'ex_date': ex,
                'payment_date': pd.Timestamp(action['payment_date']), 'announced_at': pd.NaT,
                'cash_per_share': action['cash_per_share'], 'share_factor': 1.,
                'date_evidence_basis': 'formal_action_registry',
                'evidence_sha256': sha256(ROOT / 'strategy_corporate_actions.json')}
            frames['etf_actions'] = pd.concat([frames['etf_actions'], pd.DataFrame([event])], ignore_index=True)
    manifest = {**parent.manifest, 'data_version': directory.name, 'files': [],
        'parent_manifest_path': str(parent_path.resolve()), 'parent_manifest_sha256': sha256(parent_path),
        'calendar': {'start': '2010-01-01', 'end': str(calendar[-1].date())},
        'formal_refresh_sources': sources, 'formal_refresh_errors': errors,
        'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}
    for name, frame in frames.items():
        if frame.empty:
            continue
        path = destination / f'{name}.csv'
        frame.to_csv(path, index=False)
        manifest['files'].append({'name': name, 'path': path.name, 'sha256': sha256(path), 'rows': len(frame)})
    manifest_path = destination / 'manifest.json'
    write_json_atomic(manifest_path, manifest)
    data = load_snapshot(manifest_path)
    for symbol in sorted(symbols):
        if data.etf_bars(symbol).trade_date.max() < end:
            raise ValueError(f'{symbol} 估值账户原价不能覆盖共同截止日')
    return data, errors


def valuation_row(signal, definition):
    return json_safe({'metric': definition['valuation_metric'], 'index_id': '000300' if definition['id'] == IDS[0] else '399006',
        'value': signal.get('value'), 'trade_date': signal.get('valuation_date'),
        'available_at': signal.get('valuation_available_at'), 'lag_sessions': signal.get('valuation_lag'),
        'entry_upper': signal.get('L1'), 'entry_mid': signal.get('L2'), 'entry_low': signal.get('L3'),
        'exit_level': signal.get('S'), 'exit_quantile': definition['policy']['exit_quantile'],
        'phase': signal.get('phase_before'), 'buy_steps': signal.get('buy_steps_before'),
        'reason': signal.get('reason'), 'detail': signal.get('detail')})


def build_strategy(directory, definition, spec, data, run_id, end, stats_fn, next_session_fn):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if protocol['research_start'] != spec['start'] or protocol['initial_capital'] != spec['initial_capital']:
        raise ValueError('估值策略共同起点/初始资金与正式登记不一致')
    case = next(c for c in protocol['cases'] if c['id'] == definition['case_id'])
    result = replay(data, protocol, case, end)
    if result.status != 'historical_run_complete' or result.daily.session.iloc[-1] != end:
        raise ValueError(f'{definition["id"]} 正式估值回放失败：{result.status_detail}')
    case_dir = directory / 'valuation_cases' / definition['id']
    case_dir.mkdir(parents=True)
    for name, frame in [('daily_account', result.daily), ('signals', result.signals), ('orders', result.orders),
                         ('trades', result.trades), ('actions', result.actions), ('calibration', result.calibration_rows),
                         ('episodes', result.episodes)]:
        frame.to_csv(case_dir / f'{name}.csv', index=False)
    write_json_atomic(case_dir / 'metrics.json', result.metrics)
    bars = {definition['symbol']: data.etf_bars(definition['symbol'])}
    ledger = _recompute_case(case_dir, bars, spec['initial_capital'],
        action_sources={s: data.etf_actions(s) for s in bars},
        max_price_carry_sessions=protocol['execution']['max_price_carry_sessions'])
    write_json_atomic(case_dir / 'verification.json', ledger)
    if ledger.get('status') != 'passed':
        raise ValueError(f'{definition["id"]} 估值账本独立验收失败')
    sid, symbol = definition['id'], definition['symbol']
    daily = result.daily.set_index('session')
    report = pd.DataFrame({'account': daily.nav, 'cash': daily.cash, 'value': daily.equity_value,
                           'dividend_receivable': daily.dividend_receivable})
    report.index.name = 'date'
    # 使用逐日数量与原价重建长表，不用已舍入到分的持仓市值。
    holdings = pd.DataFrame({'date': daily.index, 'symbol': symbol,
        'quantity': daily[f'quantity_{symbol}'].to_numpy(),
        'value': (daily[f'quantity_{symbol}'] * daily[f'price_{symbol}']).to_numpy()})
    trades = result.trades.copy()
    if trades.empty:
        trades = pd.DataFrame(columns=['date', 'signal_date', 'symbol', 'side', 'quantity', 'price', 'notional', 'commission', 'participation', 'reason'])
    else:
        trades['date'] = trades.execution_date
        trades['participation'] = trades.notional / trades.adv20
        # 用对应开盘原价与成交价差核算滑点/冲击金额，不用估值收盘价。
        opens = data.etf_bars(symbol).set_index('trade_date')['open']
        trades['slippage_cost'] = [float(row.quantity) * abs(float(row.price) - float(opens.loc[pd.Timestamp(row.execution_date)]))
                                   for row in trades.itertuples()]
    if len(trades) and (not (trades.participation <= .020000001).all()):
        raise ValueError('估值策略成交额参与率超限')
    # 周首/T+1核查独立于信号函数。
    from src.formal_rules import weekly_sessions
    if len(trades):
        signal_days = pd.to_datetime(trades.signal_date)
        expected = data.calendar[data.calendar.get_indexer(signal_days) + 1]
        if not signal_days.isin(weekly_sessions(report.index, spec['start'])).all() or not np.array_equal(
                expected.strftime('%Y-%m-%d'), pd.to_datetime(trades.date).dt.strftime('%Y-%m-%d')):
            raise ValueError('估值策略存在非周首/T+1交易')
    metrics, dd = stats_fn(report, spec['initial_capital'])
    latest = result.signals.iloc[-1].to_dict()
    signal_date = pd.Timestamp(latest['session'])
    signal_day = daily.loc[signal_date]
    current_day = daily.iloc[-1]
    price = float(signal_day[f'price_{symbol}'])
    quantity = int(signal_day[f'quantity_{symbol}'])
    orders = result.orders[result.orders.signal_date.eq(str(signal_date.date()))] if 'signal_date' in result.orders else pd.DataFrame()
    rows = []
    for order in orders.to_dict('records'):
        requested = order.get('requested_quantity')
        q = quantity if requested is None or pd.isna(requested) else int(requested)
        current_weight = quantity * price / float(signal_day.nav)
        target = max(0., current_weight + (1 if order['side'] == 'buy' else -1) * q * price / float(signal_day.nav))
        rows.append({'symbol': symbol, 'code': symbol[2:], 'name': universe.UNIVERSE.get(symbol[2:], symbol),
            'side': order['side'], 'quantity': q, 'reference_price': price, 'estimated_amount': q * price,
            'current_weight': current_weight,
            'target_weight': target,
            'reason': str(latest['detail'])})
    if not rows and quantity:
        rows.append({'symbol': symbol, 'code': symbol[2:], 'name': universe.UNIVERSE.get(symbol[2:], symbol),
            'side': 'hold', 'quantity': 0, 'reference_price': price, 'estimated_amount': 0.,
            'current_weight': quantity * price / float(signal_day.nav),
            'target_weight': quantity * price / float(signal_day.nav), 'reason': str(latest['detail'])})
    blocked = latest.get('valuation_lag') is None or pd.isna(latest.get('valuation_lag')) \
        or float(latest['valuation_lag']) > protocol['execution']['max_valuation_lag_sessions'] \
        or 'calibration_unavailable' in str(latest.get('skips', ''))
    valuation = valuation_row(latest, definition)
    context = {'date': str(signal_date.date()), 'prices': {symbol: price},
        'eligible': [symbol] if data.eligibility(symbol, signal_date)['eligible'] else [],
        'capabilities': {'account_preview': False, 'reason': PREVIEW_REASON}, 'valuation': valuation}
    raw_end = data.index_valuation(valuation['index_id'], definition['valuation_metric']).trade_date.max()
    payload = {'id': sid, 'name': definition['name'], 'color': definition['color'],
        'kind': 'valuation_bands', 'status': 'formal_service', 'description': definition['description'],
        'version': spec['version'], 'run_id': run_id, 'data_start': spec['start'], 'data_end': str(end.date()),
        'account_basis': definition['account_basis'], 'initial_capital': spec['initial_capital'],
        'metrics': metrics, 'goals': {'cagr': metrics['cagr'] >= spec['goals']['cagr'],
            'worst_complete_year': bool(metrics['worst_complete_year'] and metrics['worst_complete_year']['return'] >= spec['goals']['worst_complete_year']),
            'max_drawdown': metrics['max_drawdown'] <= spec['goals']['max_drawdown']},
        'investment_admission': False, 'valuation': valuation, 'valuation_data_end': str(raw_end.date()),
        'valuation_basis': 'ETF.run等权估值；历史重建非PIT' + ('；PB财务MRQ口径未经证实' if sid == IDS[1] else ''),
        'limitations': ['等权估值，非原整体法；历史重建非PIT', *(['PB财务MRQ口径未经证实'] if sid == IDS[1] else []),
            '历史分红含研究假设；仅少量闭合周期；年度亏损及年化目标未通过'],
        'capabilities': context['capabilities'], 'cash': float(current_day.cash),
        'dividend_receivable': float(current_day.dividend_receivable),
        'holdings': [{'symbol': symbol, 'code': symbol[2:], 'name': universe.UNIVERSE.get(symbol[2:], symbol),
            'quantity': int(current_day[f'quantity_{symbol}']), 'price': float(current_day[f'price_{symbol}']),
            'value': float(current_day.equity_value), 'weight': float(current_day.equity_value/current_day.nav)}]
            if current_day[f'quantity_{symbol}'] else [],
        'advice': {'signal_date': str(signal_date.date()), 'execution_date': str(next_session_fn(signal_date).date()),
            'account': float(signal_day.nav), 'cash': float(signal_day.cash), 'rows': rows, 'orders_count': len(orders),
            'blocked': bool(blocked), 'blocked_reason': '估值或年度校准数据不可用，暂停新增建议' if blocked else None,
            'basis': '模型账户历史批次与冷却状态；周首19:00评估、下一交易日开盘模拟执行',
            'decision_detail': latest['detail']},
        'series': {'dates': report.index.strftime('%Y-%m-%d').tolist(), 'capital': report.account.tolist(),
                   'drawdown': (-dd).tolist(), 'cash': report.cash.tolist()},
        'weekly_holdings': weekly_holdings_history(report, holdings),
        'weekly_allocation': weekly_holdings_history(report, holdings, weeks=50),
        'validation': {'weekly_t_plus_1': True, 'ledger': ledger, 'trade_count': len(trades),
                       'issues': result.data_failures}}
    write_json_atomic(directory / f'{sid}.json', payload)
    write_json_atomic(directory / f'{sid}_context.json', context)
    signals = result.signals.copy()
    signals['etf_close'] = [float(daily.loc[pd.Timestamp(day), f'price_{symbol}']) for day in signals.session]
    signals['eligible'] = [bool(data.eligibility(symbol, day)['eligible']) for day in signals.session]
    signals.to_csv(directory / f'{sid}_signals.csv', index=False)
    report.to_csv(directory / f'{sid}_report.csv')
    holdings.to_csv(directory / f'{sid}_holdings.csv', index=False)
    trades.to_csv(directory / f'{sid}_trades.csv', index=False)
    return payload
