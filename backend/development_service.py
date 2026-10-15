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
from src.formal_rules import SYMBOLS, weekly_sessions
from src.formal_service import ROOT, OUT, DATABASE, runtime_status
from src.market import load_market
from src.runs import sha256

RUN_PATTERN = re.compile(r'^\d{8}-\d{6}-[a-f0-9]{8}$')
STRATEGY_IDS = ('lightgbm', 'trend200_vol15')
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
        required = ['registration.json', 'folds.json', 'predictions_lightgbm.pkl']
        for id_ in STRATEGY_IDS:
            required += [id_ + '.json', id_ + '_context.json', id_ + '_trades.csv']
        # 每次请求校验；不可变文件损坏后不能继续返回旧缓存。
        expected = completed['artifact_hashes']
        for name in required:
            if sha256(directory / name) != expected[name]:
                raise ValueError('正式产物哈希校验失败：' + name)
        cache_key = (str(directory), tuple((name, expected[name]) for name in required))
        with _lock:
            if cache_key not in _cache:
                registration = _read_json(directory / 'registration.json')
                definitions = registration['registry']['strategies']
                if {d['id'] for d in definitions} != set(STRATEGY_IDS):
                    raise ValueError('登记必须包含两条正式策略')
                payloads = [_read_json(directory / (id_ + '.json')) for id_ in STRATEGY_IDS]
                ends = {item['data_end'] for item in payloads}
                if any(item['run_id'] != selected for item in payloads) or len(ends) != 1:
                    raise ValueError('策略快照混用了运行或行情截止日')
                data_end = next(iter(ends))
                if registration['data_audit']['end'] != data_end:
                    raise ValueError('数据审计截止日与策略快照不一致')
                predictions = pd.read_pickle(directory / 'predictions_lightgbm.pkl')
                if not isinstance(predictions, pd.Series) or not isinstance(predictions.index, pd.MultiIndex):
                    raise ValueError('正式预测格式不合法')
                if predictions.index.names != ['datetime', 'instrument'] or predictions.index.has_duplicates:
                    raise ValueError('正式预测索引不合法')
                _cache[cache_key] = {
                    'run_id': selected, 'directory': directory, 'completed_at': completed['completed_at'],
                    'data_end': data_end, 'refresh_error': completed.get('refresh_error'),
                    'strategies': payloads, 'registration': registration,
                    'folds': _read_json(directory / 'folds.json')['folds'], 'predictions': predictions,
                    'contexts': {id_: _read_json(directory / (id_ + '_context.json')) for id_ in STRATEGY_IDS},
                    'trades': {id_: json_safe(pd.read_csv(directory / (id_ + '_trades.csv')).to_dict('records'))
                               for id_ in STRATEGY_IDS},
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
    payload = copy.deepcopy(next(s for s in bundle['strategies'] if s['id'] == strategy_id))
    payload['runtime'] = runtime_status(payload)
    return payload


def overview(run_id=None):
    bundle = _bundle(run_id)
    return {
        'source': 'formal_weekly', 'run_id': bundle['run_id'],
        'completed_at': bundle['completed_at'], 'data_end': bundle['data_end'],
        'refresh_error': bundle['refresh_error'],
        'strategies': [_payload(bundle, id_) for id_ in STRATEGY_IDS],
        'registry': copy.deepcopy(bundle['registration']['registry']),
        'data_audit': copy.deepcopy(bundle['registration']['data_audit']),
        'folds': copy.deepcopy(bundle['folds']),
    }


def _definition(bundle, strategy_id):
    return next(d for d in bundle['registration']['registry']['strategies'] if d['id'] == strategy_id)


def _symbols(bundle, strategy_id):
    if strategy_id == 'lightgbm':
        return sorted(SYMBOLS)
    definition = _definition(bundle, strategy_id)
    return sorted(set(definition['candidate']['weights']) | {
        s for alternatives in definition['candidate'].get('fallbacks', {}).values() for s in alternatives})


def get_universe(strategy_id, run_id=None):
    _strategy_id(strategy_id)
    bundle = _bundle(run_id)
    context = bundle['contexts'][strategy_id]
    eligible = set(context['eligible'])
    return {'run_id': bundle['run_id'], 'date': context['date'], 'rows': [
        {'symbol': s, 'code': s[2:], 'name': universe.UNIVERSE.get(s[2:], s),
         'asset_class': universe.get_asset_class(s[2:]),
         'eligible': s in eligible and finite_float(context['prices'].get(s)) is not None}
        for s in _symbols(bundle, strategy_id)]}


def _dates(bundle, strategy_id):
    payload = next(s for s in bundle['strategies'] if s['id'] == strategy_id)
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


def _rank_rows(bundle, market, date, topk):
    definition = _definition(bundle, 'lightgbm')
    cores = definition['policy']['defensive_weights']
    allowed = market.eligibility[market.eligibility.date.eq(date)].set_index('symbol')['eligible']
    close = market.wide('close').loc[date]
    predictions = bundle['predictions']
    try:
        scores = predictions.xs(date, level='datetime').dropna()
    except KeyError:
        scores = pd.Series(dtype=float)
    satellites = set(_symbols(bundle, 'lightgbm')) - set(cores)
    scores = scores[[s in satellites and bool(allowed.get(s, False)) and
                     universe.get_asset_class(s[2:]) == 'equity' for s in scores.index]]
    # 相同预测分数按代码稳定排序，不给未预测资产编造分数。
    ranking = pd.DataFrame({'symbol': scores.index, 'score': scores.values,
                            'rank': scores.rank(ascending=False, method='min').values}).sort_values(
        ['score', 'symbol'], ascending=[False, True]).head(topk)
    rows = []
    for _, item in ranking.iterrows():
        s = item['symbol']
        row = _base_row(s, True, close.get(s))
        row.update(score=float(item['score']), rank=int(item['rank']), reason='当年登记模型真实预测；已通过当周市场资格')
        rows.append(row)
    core_rows = []
    for s, weight in cores.items():
        row = _base_row(s, allowed.get(s, False), close.get(s))
        row.update(candidate_weight=weight, reason='核心底仓；不参与权益卫星预测排名' if row['eligible'] else
                   '核心底仓当周市场资格不合格，相应预算留现金')
        core_rows.append(row)
    return rows, core_rows


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
    market = _market(bundle)
    if selected not in market.calendar:
        raise DevelopmentUnavailableError('正式信号日缺少登记市场价格')
    if strategy_id == 'lightgbm':
        rows, core_rows = _rank_rows(bundle, market, selected, topk)
        description = '正式年度滚动 LightGBM 的当周权益卫星排名；核心底仓单列。同分共享名次，代码顺序仅用于展示；排名不等于最终调仓目标。'
    else:
        rows, core_rows = _trend_rows(bundle, market, selected), []
        description = '固定通信/黄金策略没有预测分数；逐标的展示真实资格与200日复权趋势，沪深300仅作资格替代。'
    return json_safe({'source': 'formal_weekly', 'run_id': bundle['run_id'], 'strategy_id': strategy_id,
                      'date': str(selected.date()), 'requested_date': requested, 'fallback': bool(fallback),
                      'kind': 'rank' if strategy_id == 'lightgbm' else 'trend', 'rows': rows,
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
            'folds': copy.deepcopy(bundle['folds']) if strategy_id == 'lightgbm' else [],
            'validation': copy.deepcopy(payload['validation']),
            'definition': copy.deepcopy(_definition(bundle, strategy_id))}
