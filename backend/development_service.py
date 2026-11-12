"""开发后台的正式周频产物查询；只读绑定运行，不接旧研究指针。"""
from __future__ import annotations

import copy
import json
import re
import threading

import duckdb
import pandas as pd
import universe
from src.artifacts import finite_float, json_safe
from src.formal_rules import SYMBOLS, weekly_sessions, ACTIVE_STRATEGY_IDS
from src.formal_service import ROOT, OUT, DATABASE, runtime_status
from src.market import load_market
from src.runs import sha256

RUN_PATTERN = re.compile(r'^\d{8}-\d{6}-[a-f0-9]{8}$')
STRATEGY_IDS = ACTIVE_STRATEGY_IDS
_lock = threading.RLock()
_cache = {}
_markets = {}


class DevelopmentUnavailableError(RuntimeError):
    """正式运行未发布、损坏或输入无法与登记快照核对。"""


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _strategy_id(strategy_id):
    if strategy_id not in STRATEGY_IDS:
        raise KeyError(strategy_id)
    return strategy_id


def _bundle(run_id=None):
    """先固定一次指针，后续所有文件均从同一不可变目录读取。"""
    try:
        pointer = _read_json(OUT / 'current.json') if run_id is None else None
        selected = pointer['run_id'] if pointer else run_id
        if not isinstance(selected, str) or not RUN_PATTERN.fullmatch(selected):
            raise ValueError('运行编号不合法')
        directory = (OUT / 'runs' / selected).resolve()
        if directory.parent != (OUT / 'runs').resolve():
            raise ValueError('正式运行目录不合法')
        if pointer and (ROOT / pointer['directory']).resolve() != directory:
            raise ValueError('正式指针目录与运行编号不一致')
        completed = _read_json(directory / 'completion.json')
        if completed['run_id'] != selected:
            raise ValueError('完成记录与运行编号不一致')
        required = ['registration.json']
        expected = completed['artifact_hashes']
        if sha256(directory / 'registration.json') != expected['registration.json']:
            raise ValueError('正式登记文件哈希校验失败')
        registration = _read_json(directory / 'registration.json')
        definitions = registration['registry']['strategies']
        ids = [d['id'] for d in definitions]
        if tuple(ids) != STRATEGY_IDS:
            raise ValueError('该运行不属于当前三项策略，停用策略不再加载，请重建快照')
        for id_ in ids:
            required += [id_ + '.json', id_ + '_context.json', id_ + '_trades.csv']
        for definition in definitions:
            if definition.get('kind') == 'valuation_bands':
                required.append(definition['id'] + '_signals.csv')
            elif definition.get('kind') == 'index_benchmark':
                required += ['index_data/H20269.csv', 'index_data/manifest.json']
            elif definition.get('kind') == 'dividend_band':
                required += [definition['id'] + '_signals.csv', definition['id'] + '_daily.csv',
                             'dividend_data/panel.csv', 'dividend_data/manifest.json']
        # 每次请求校验；不可变文件损坏后不能继续返回旧缓存。
        for name in required:
            if sha256(directory / name) != expected[name]:
                raise ValueError('正式产物哈希校验失败：' + name)
        cache_key = (str(directory), tuple((name, expected[name]) for name in required))
        with _lock:
            if cache_key not in _cache:
                payloads = [_read_json(directory / (id_ + '.json')) for id_ in ids]
                ends = {item['data_end'] for item in payloads}
                if any(item['run_id'] != selected for item in payloads) or len(ends) != 1:
                    raise ValueError('策略快照混用了运行或行情截止日')
                data_end = next(iter(ends))
                if registration['data_audit']['end'] != data_end:
                    raise ValueError('数据审计截止日与策略快照不一致')
                _cache[cache_key] = {
                    'run_id': selected, 'directory': directory, 'completed_at': completed['completed_at'],
                    'data_end': data_end, 'refresh_error': completed.get('refresh_error'),
                    'strategies': payloads, 'registration': registration,
                    'folds': [],
                    'contexts': {id_: _read_json(directory / (id_ + '_context.json')) for id_ in ids},
                    'trades': {id_: json_safe(pd.read_csv(directory / (id_ + '_trades.csv')).to_dict('records'))
                               for id_ in ids},
                }
                # 保留上一运行供已经打开的页面完成请求；无界版本缓存不必要。
                while len(_cache) > 3:
                    _cache.pop(next(iter(_cache)))
            result = _cache[cache_key]
        if pointer and pointer['data_end'] != result['data_end']:
            raise ValueError('正式指针行情截止日不一致')
        return result
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise DevelopmentUnavailableError('正式周频开发数据不可用：' + str(exc)) from exc


def _payload(bundle, strategy_id):
    item = next((s for s in bundle['strategies'] if s['id'] == strategy_id), None)
    if item is None:
        raise KeyError(strategy_id)
    payload = copy.deepcopy(item)
    payload['runtime'] = runtime_status(payload)
    return payload


def overview(run_id=None):
    bundle = _bundle(run_id)
    return {
        'source': 'formal_weekly', 'run_id': bundle['run_id'],
        'completed_at': bundle['completed_at'], 'data_end': bundle['data_end'],
        'refresh_error': bundle['refresh_error'],
        'strategies': [_payload(bundle, item['id']) for item in bundle['strategies']],
        'registry': copy.deepcopy(bundle['registration']['registry']),
        'data_audit': copy.deepcopy(bundle['registration']['data_audit']),
        'folds': copy.deepcopy(bundle['folds']),
    }


def _definition(bundle, strategy_id):
    definition = next((d for d in bundle['registration']['registry']['strategies'] if d['id'] == strategy_id), None)
    if definition is None:
        raise KeyError(strategy_id)
    return definition


def _symbols(bundle, strategy_id):
    definition = _definition(bundle, strategy_id)
    if definition.get('kind') == 'index_benchmark':
        return []
    if definition.get('kind') == 'dividend_band':
        return ['H20269', 'SH511260']
    if definition.get('kind') == 'valuation_bands':
        return [definition['symbol']]
    return sorted(set(definition['candidate']['weights']) | {
        s for alternatives in definition['candidate'].get('fallbacks', {}).values() for s in alternatives})


def get_universe(strategy_id, run_id=None):
    _strategy_id(strategy_id)
    bundle = _bundle(run_id)
    context = bundle['contexts'][strategy_id]
    if _definition(bundle, strategy_id).get('kind') == 'dividend_band':
        return {'run_id': bundle['run_id'], 'date': context['date'], 'kind': 'dividend_band',
                'rows': [{'symbol': symbol, 'code': symbol, 'name': name, 'eligible': False,
                          'tradeable': False, 'equivalent': True, 'asset_class': asset}
                         for symbol, name, asset in [('H20269','红利低波（指数等值）','equity'),
                                                     ('SH511260','十年国债（复权等值）','bond')]]}
    eligible = set(context['eligible'])
    return {'run_id': bundle['run_id'], 'date': context['date'], 'rows': [
        {'symbol': s, 'code': s[2:], 'name': universe.UNIVERSE.get(s[2:], s),
         'asset_class': universe.get_asset_class(s[2:]),
         'eligible': s in eligible and finite_float(context['prices'].get(s)) is not None}
        for s in _symbols(bundle, strategy_id)]}


def _dates(bundle, strategy_id):
    payload = _payload(bundle, strategy_id)
    calendar = pd.DatetimeIndex(pd.to_datetime(payload['series']['dates']))
    return weekly_sessions(calendar, bundle['registration']['registry']['start'])


def available_dates(strategy_id, run_id=None):
    _strategy_id(strategy_id)
    bundle = _bundle(run_id)
    return {'run_id': bundle['run_id'], 'dates': _dates(bundle, strategy_id).strftime('%Y-%m-%d').tolist()}


def _market(bundle):
    """截止登记日期重建因果资格；历史修订不能静默替换已发布输入。"""
    fingerprint = bundle['registration']['market_fingerprint']
    key = (str(DATABASE), bundle['data_end'], fingerprint)
    with _lock:
        if key not in _markets:
            try:
                market = load_market(DATABASE, as_of=bundle['data_end'])
                if market.fingerprint != fingerprint:
                    raise ValueError('当前数据库的登记历史已改变，不能回填该运行；请发布新的正式快照')
                _markets[key] = market
                while len(_markets) > 3:
                    _markets.pop(next(iter(_markets)))
            except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
                raise DevelopmentUnavailableError('正式历史输入核对失败：' + str(exc)) from exc
        return _markets[key]


def _date(bundle, strategy_id, requested):
    dates = _dates(bundle, strategy_id)
    if not len(dates):
        raise DevelopmentUnavailableError('该正式运行没有周频信号日')
    if requested is None:
        return dates[-1], None, False
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', requested):
            raise ValueError()
        wanted = pd.Timestamp(requested)
    except (ValueError, TypeError):
        raise ValueError('日期必须是有效的 YYYY-MM-DD 日期')
    available = dates[dates <= wanted]
    if not len(available):
        raise ValueError('所选日期早于正式回测起点')
    actual = available[-1]
    return actual, requested, actual != wanted


def _base_row(symbol, eligible, close):
    return {'symbol': symbol, 'code': symbol[2:], 'name': universe.UNIVERSE.get(symbol[2:], symbol),
            'eligible': bool(eligible), 'score': None, 'rank': None, 'trend_pass': None,
            'close': finite_float(close), 'sma200': None, 'deviation': None, 'reason': ''}


def _trend_rows(bundle, market, date):
    definition = _definition(bundle, 'trend200_vol15')
    candidate = definition['candidate']
    allowed = market.eligibility[market.eligibility.date.eq(date)].set_index('symbol')['eligible']
    close = market.wide('close').loc[date]
    history = market.wide('adj_close').loc[:date].tail(definition.get('trend_window', 200))
    fallback_from = {replacement: original for original, alternatives in candidate.get('fallbacks', {}).items()
                     for replacement in alternatives}
    rows = []
    for s in _symbols(bundle, 'trend200_vol15'):
        values = history[s] if s in history else pd.Series(dtype=float)
        complete = len(values) == 200 and values.notna().all() and values.gt(0).all()
        last = finite_float(values.iloc[-1]) if len(values) else None
        mean = float(values.mean()) if complete else None
        row = _base_row(s, allowed.get(s, False), close.get(s))
        passed = bool(complete and last > mean)
        raw_price = row['close']
        row.update(trend_pass=passed,
                   sma200=mean * raw_price / last if mean is not None and last and raw_price else None,
                   deviation=last / mean - 1 if mean else None)
        original = fallback_from.get(s)
        active = not bool(allowed.get(original, False)) if original else True
        weight = candidate['weights'].get(s, 0.)
        if original:
            weight = candidate['weights'][original] * candidate['fallbacks'][original][s]
        row['fallback_active'] = active if original else False
        row['candidate_weight'] = float(weight if active and row['eligible'] and passed else 0.)
        if not row['eligible']:
            reason = '当周市场资格不合格；该标的预算留现金'
            if s in candidate.get('fallbacks', {}):
                reason += '，允许检查沪深300资格和趋势后替代'
        elif original and not active:
            reason = '通信市场资格合格，替代条件未触发；通信跌破均线也不会启用沪深300'
        elif not complete:
            reason = '不足200个有效交易日价格，趋势门槛不满足，预算留现金'
        elif not passed:
            reason = '复权收盘价未严格高于200日均线，预算留现金'
        else:
            reason = '市场资格与200日趋势门槛均通过；仍需波动、漂移与成交约束'
        row['reason'] = reason
        rows.append(row)
    return rows


def get_picks(strategy_id, date=None, topk=29, run_id=None):
    _strategy_id(strategy_id)
    if not isinstance(topk, int) or not 1 <= topk <= 29:
        raise ValueError('展示数量需在1至29之间')
    bundle = _bundle(run_id)
    selected, requested, fallback = _date(bundle, strategy_id, date)
    definition = _definition(bundle, strategy_id)
    if definition.get('kind') == 'dividend_band':
        from src.formal_dividend import signal_detail, holdings_rows, allocation_advice
        signals = pd.read_csv(bundle['directory'] / f'{strategy_id}_signals.csv', index_col='session', parse_dates=True)
        daily = pd.read_csv(bundle['directory'] / f'{strategy_id}_daily.csv', index_col='date', parse_dates=True)
        if selected not in signals.index or selected not in daily.index:
            raise DevelopmentUnavailableError('A策略历史信号或账户日期缺失')
        signal, account = signals.loc[selected], daily.loc[selected]
        if not isinstance(signal, pd.Series) or not isinstance(account, pd.Series):
            raise DevelopmentUnavailableError('A策略历史信号或账户日期重复')
        payload = _payload(bundle, strategy_id)
        return json_safe({'source': 'formal_weekly', 'run_id': bundle['run_id'],
            'strategy_id': strategy_id, 'date': str(selected.date()), 'requested_date': requested,
            'fallback': bool(fallback), 'kind': 'dividend_band',
            'dividend_signal': signal_detail(signal, selected),
            'rows': allocation_advice(account, signal, definition), 'core_rows': [],
            'holdings': holdings_rows(account), 'cash': float(account.cash), 'account': float(account.nav),
            'capabilities': payload['capabilities'], 'description': 'A隐含股息率七档估值带；仅展示模型等值配置，不生成ETF整数委托'})
    if definition.get('kind') == 'index_benchmark':
        payload = _payload(bundle, strategy_id)
        frame = pd.read_csv(bundle['directory'] / 'index_data/H20269.csv', index_col='date', parse_dates=True)
        benchmark = {**payload['benchmark'], 'close': float(frame.loc[selected, 'close']),
                     'data_date': str(selected.date())}
        return {'source': 'formal_reference', 'run_id': bundle['run_id'], 'strategy_id': strategy_id,
                'date': str(selected.date()), 'requested_date': requested, 'fallback': bool(fallback),
                'kind': 'index_benchmark', 'rows': [], 'core_rows': [],
                'benchmark': benchmark,
                'description': '100%红利低波全收益指数参考，不产生ETF候选、预测排名或调仓指令。'}
    if definition.get('kind') == 'valuation_bands':
        from src.formal_valuation import valuation_row
        signals = pd.read_csv(bundle['directory'] / f'{strategy_id}_signals.csv')
        matching = signals[signals.session.eq(str(selected.date()))]
        if len(matching) != 1:
            raise DevelopmentUnavailableError('正式估值信号记录缺失或重复')
        signal = matching.iloc[0].to_dict()
        row = _base_row(definition['symbol'], signal['eligible'], signal['etf_close'])
        row.update(valuation=valuation_row(signal, definition), reason=signal['detail'])
        return json_safe({'source': 'formal_weekly', 'run_id': bundle['run_id'], 'strategy_id': strategy_id,
            'date': str(selected.date()), 'requested_date': requested, 'fallback': bool(fallback),
            'kind': 'valuation', 'rows': [row], 'core_rows': [], 'valuation': row['valuation'],
            'description': '等权估值分批策略：过去五个完整年低点校准，按周评估；历史重建非PIT，PB财务MRQ口径未证实。'})
    market = _market(bundle)
    if selected not in market.calendar:
        raise DevelopmentUnavailableError('正式信号日缺少登记市场价格')
    rows, core_rows = _trend_rows(bundle, market, selected), []
    description = '固定通信/黄金策略没有预测分数；逐标的展示真实资格与200日复权趋势，沪深300仅作资格替代。'
    return json_safe({'source': 'formal_weekly', 'run_id': bundle['run_id'], 'strategy_id': strategy_id,
                      'date': str(selected.date()), 'requested_date': requested, 'fallback': bool(fallback),
                      'kind': 'trend', 'rows': rows,
                      'core_rows': core_rows, 'description': description})


def backtest(strategy_id, run_id=None):
    _strategy_id(strategy_id)
    bundle = _bundle(run_id)
    payload = _payload(bundle, strategy_id)
    trades = [{**row, 'code': row['symbol'][2:],
               'name': universe.UNIVERSE.get(row['symbol'][2:], row['symbol'])}
              for row in bundle['trades'][strategy_id]]
    return {'source': 'formal_weekly', 'run_id': bundle['run_id'], 'strategy': payload,
            'trades': trades,
            'folds': [],
            'validation': copy.deepcopy(payload['validation']),
            'definition': copy.deepcopy(_definition(bundle, strategy_id))}
