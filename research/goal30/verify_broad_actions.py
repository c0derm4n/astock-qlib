"""独立核对固定事件范围、原始公告索引、PDF重提取及未修订的行情偏差。"""
from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.broad_action_evidence import ROOT, READINESS, compact, parse_sources
from research.goal30.bond_dividend_evidence import digest
from research.goal30.turnover_budget import require

PRIOR = BASE / 'output/goal30/daily_ledger_validation/20260913-235145-0e2ea218/registration.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def event_key(event):
    return '_'.join((event['symbol'], event['type'], event.get('ex_date', event.get('effective_date'))))


def factor_diagnostic(before, after, ratio):
    """诊断已保存因子；仅以四位小数舍入的可解释区间作交叉核对。"""
    before_factor, after_factor = (Decimal(str(row['adj_factor'])) for row in (before, after))
    ratio = Decimal(ratio)
    require(before_factor > 0 and after_factor > 0, '复权因子无效')
    quantum = Decimal('0.00005')
    require(before_factor > quantum, '复权因子过小，四位小数诊断不适用')
    low = (after_factor - quantum) / (before_factor + quantum)
    high = (after_factor + quantum) / (before_factor - quantum)
    old_return = float(after['adj_close'] / before['adj_close'] - 1)
    share_return = float(after['close'] * float(ratio) / before['close'] - 1)
    return {'factor_before': float(before_factor), 'factor_after': float(after_factor),
        'stored_factor_jump': float(after_factor / before_factor), 'official_ratio': str(ratio),
        'four_decimal_rounding_envelope': [float(low), float(high)],
        'outside_rounding_envelope': not low <= ratio <= high,
        'stored_adjusted_close_return': old_return,
        'official_split_only_close_return_before_holder_rounding': share_return,
        'return_difference_percentage_points': (old_return - share_return) * 100}


def verify():
    from pypdf import PdfReader
    inputs = {}

    def checked(path, expected=None):
        path = Path(path)
        actual = digest(path)
        require(expected is None or actual == expected, f'输入指纹改变：{path}')
        inputs[path.relative_to(BASE).as_posix()] = actual
        return path

    scope = read(checked(ROOT / 'scope.json'))
    readiness = read(checked(READINESS, scope['readiness_sha256']))
    sources = read(checked(ROOT / 'sources.json'))['sources']
    visual = read(checked(ROOT / 'visual_review.json'))
    scope_by_id = {event_key(e): e for e in scope['events']}
    require(len(scope_by_id) == len(scope['events']) == 20 and
            len({e['symbol'] for e in scope['events']}) == 13, '事件范围不同')
    for event in scope['events']:
        source_event = {k: v for k, v in event.items() if k != 'symbol'}
        require(source_event in readiness['symbols'][event['symbol']]['sina_events_in_tradable_window'],
                '固定范围不是上一阶段已知新浪事件')

    indexes = {}
    for metadata in sorted(ROOT.glob('*_index_metadata.json')):
        meta = read(checked(metadata))
        raw = checked(ROOT / metadata.name.replace('_metadata.json', '.bin'), meta['sha256'])
        require(meta['status'] == 200, '索引请求未成功')
        if 'sse.com.cn' in meta['url']:
            index = read(raw)
            require(index['pageHelp']['pageCount'] == 1 and
                    index['pageHelp']['total'] == len(index['result']), '交易所窗口检索未完成分页')
        elif 'cninfo.com.cn' in meta['url']:
            index = read(raw)
            require(index['hasMore'] is False and index['totalAnnouncement'] == len(index['announcements']),
                    '巨潮窗口检索未完成分页')
        else:
            continue  # 新浪转载索引只作为归档旁证，不称发行人目录完整。
        indexes[raw.name] = index
    primary = [s for s in sources if s['source_kind'].startswith('issuer_pdf')]
    require(len(primary) == 32 and len(indexes) == 20, '原PDF或官方窗口索引数量不符')
    for source in sources:
        checked(ROOT / source['file'], source['sha256'])
        require(source['status'] == 200 and source['event_key'] in scope_by_id, '来源状态或范围不同')
        text_file = ROOT / Path(source['file']).with_suffix('.txt')
        if text_file.exists():
            checked(text_file)
        if source not in primary:
            continue
        index = indexes[source['index_file']]
        if source['source_kind'] == 'issuer_pdf_on_sse':
            matched = [r for r in index['result'] if 'https://www.sse.com.cn' + r['URL'] == source['url']
                and r['SECURITY_CODE'] == source['symbol'][2:]
                and r['SSEDATE'] == source['announcement_date'] and compact(r['TITLE']) == compact(source['title'])]
        else:
            matched = [r for r in index['announcements'] if 'https://static.cninfo.com.cn/' + r['adjunctUrl'] == source['url']
                and r['secCode'] == source['symbol'][2:]
                and str(pd.Timestamp(r['announcementTime'], unit='ms', tz='UTC').tz_convert('Asia/Shanghai').date()) == source['announcement_date']
                and compact(r['announcementTitle']) == compact(source['title'])]
        require(len(matched) == 1, 'PDF不能唯一对应原始公告索引')
        reader = PdfReader(ROOT / source['file'])
        require(len(reader.pages) == source['pages'], 'PDF页数不同')
        actual_text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        require(compact(actual_text) == compact(text_file.read_text(encoding='utf-8')), 'PDF与归档文本不同')

    for filename, review in visual['files'].items():
        checked(ROOT / filename, review['pdf_sha256'])
        source = next(s for s in primary if s['file'] == filename)
        require(review['total_pages'] == source['pages'] and
                all(1 <= p <= source['pages'] for p in review['pages_reviewed']), '目视页范围无效')
        require(len(review['rendered_hashes']) == len(review['pages_reviewed']), '目视图片计数不同')
        for page in review['pages_reviewed']:
            png = f'rendered/{Path(filename).stem}-{page}.png'
            checked(ROOT / png, review['rendered_hashes'][png])
    require(len(visual['files']) == 22 and sum(len(r['pages_reviewed']) for r in visual['files'].values()) == visual['pages_reviewed'] == 37,
            '目视记录数量不同')
    for source in primary:
        if source['kind'] != 'plan' or source['symbol'] in ('SH512010', 'SZ159928'):
            require(source['file'] in visual['files'], '所用字段原件缺少完整相关页目视')

    parsed = parse_sources()
    archived = read(checked(ROOT / 'primary_events.json'))
    require(parsed == archived['events'] and archived['investment_admission'] is False, '原PDF重提取与结果不同')
    require({p['event']['event_id'] for p in parsed} == set(scope_by_id), '主文档未完整覆盖固定20项事件')
    reconciled = 0
    for symbol, old_item in readiness['symbols'].items():
        for problem in old_item['unresolved']:
            old_event = problem['event']
            old_day = old_event.get('ex_date', old_event.get('effective_date', old_event.get('book_date')))
            matches = [p for p in parsed if p['event']['symbol'] == symbol and
                p['event']['type'] == old_event['type'] and old_day in
                {p['event'].get(k) for k in ('record_date', 'book_date', 'new_units_available_date', 'ex_date')}]
            require(len(matches) == 1, '上一清单待核对项未唯一对应原件，不能提升字段状态')
            reconciled += 1
    raw_path = next(k for k in readiness['input_hashes'] if k.endswith('market_snapshot.parquet'))
    bars = pd.read_parquet(checked(BASE / raw_path, readiness['input_hashes'][raw_path]))
    calendar = pd.DatetimeIndex(sorted(bars.date.unique()))
    comparisons, market_checks, windows = [], [], []
    for item in parsed:
        event = item['event']
        old = scope_by_id[event['event_id']]
        symbol = event['symbol']
        comparison = {'event_id': event['event_id'], 'symbol': symbol, 'sina': old, 'primary': event}
        em = readiness['symbols'][symbol]['eastmoney_events_in_tradable_window'] or []
        if event['type'] == 'split':
            dates = {event['record_date'], event['book_date'], event['new_units_available_date']}
            require(old['effective_date'] in dates, '拆分无法对应已知行动日期')
            comparison['sina_differences'] = {}
            if old['effective_date'] != event['new_units_available_date']:
                comparison['sina_differences']['effective_date'] = {'old': old['effective_date'], 'new_price_date': event['new_units_available_date']}
            if abs(Decimal(str(old['ratio'])) - Decimal(event['ratio_decimal'])) > Decimal('1e-12'):
                comparison['sina_differences']['ratio'] = {'old': old['ratio'], 'official_decimal': event['ratio_decimal']}
            paired = [e for e in em if e['type'] == 'split' and e.get('book_date', e.get('effective_date')) in dates]
            comparison['eastmoney'] = paired
            comparison['eastmoney_ratio_differences'] = [e for e in paired if abs(Decimal(str(e['ratio'])) - Decimal(event['ratio_decimal'])) > Decimal('1e-12')]
            frame = bars[bars.symbol.eq(symbol)].sort_values('date')
            new_date = pd.Timestamp(event['new_units_available_date'])
            after_book = frame[frame.date > event['book_date']]
            require(not after_book.empty and after_book.iloc[0].date == new_date, '首个新单位行情日与公告可查询日不符')
            before_rows = frame[frame.date < new_date]
            require(not before_rows.empty, '缺少折算前行情')
            before, after = before_rows.iloc[-1], after_book.iloc[0]
            require(before.date <= pd.Timestamp(event['book_date']) and
                    pd.Timestamp(event['record_date']) in calendar, '拆分登记或行情时间边界错误')
            diagnostic = factor_diagnostic(before, after, event['ratio_decimal'])
            market_checks.append({'event_id': event['event_id'], 'symbol': symbol,
                'previous_price_date': str(before.date.date()), 'first_new_price_date': str(new_date.date()),
                'availability_and_first_new_price_date_agree': True, **diagnostic})
            window = frame[frame.date.between(pd.Timestamp(event['record_date']) - pd.Timedelta(days=7), new_date + pd.Timedelta(days=3))]
            for row in window[['date', 'symbol', 'open', 'close', 'adj_factor', 'adj_close']].to_dict('records'):
                row['date'] = str(row['date'].date())
                windows.append({'event_id': event['event_id'], **row})
        else:
            require(old['ex_date'] == event['ex_date'] and abs(Decimal(str(old['cash_per_share'])) - Decimal(event['cash_per_share_decimal'])) < Decimal('1e-12'),
                    '分红日期或金额与已知序列不同')
            comparison['eastmoney'] = [e for e in em if e['type'] == 'cash_dividend' and e['ex_date'] == event['ex_date']]
            require(event['actual_cash_available_date'] is None and event['actual_receipt_confirmed'] is False,
                    '把公告日程当成真实到账')
            require(pd.Timestamp(event['ex_date']) in calendar and pd.Timestamp(event['record_date']) in calendar,
                    '场内分红登记或除息非交易日')
        comparisons.append(comparison)

    # 旧阶段登记包括已冻结代码、策略、事件、行情和正式指针。
    prior = read(checked(PRIOR))
    for path, expected in prior['input_hashes'].items():
        checked(BASE / path, expected)
    for path, expected in readiness['input_hashes'].items():
        checked(BASE / path, expected)
    for name in ('broad_action_evidence.py', 'test_broad_action_evidence.py', 'verify_broad_actions.py'):
        checked(BASE / 'research/goal30' / name)

    statuses, remaining_secondary = {}, []
    covered_symbols = {item['event']['symbol'] for item in parsed}
    for symbol, old in readiness['symbols'].items():
        secondary = [e for e in old['known_schedules'] if e['field_evidence'] != 'existing_or_new_primary_notice']
        remaining_secondary.extend({'symbol': symbol, **e} for e in secondary)
        if symbol in covered_symbols:
            statuses[symbol] = 'known_schedules_secondary_remaining' if secondary else 'known_action_fields_primary_verified'
        elif old['status'] == 'known_dividend_schedule_secondary_only':
            statuses[symbol] = 'known_schedules_secondary_remaining'
        else:
            statuses[symbol] = old['status']
    require('known_action_fields_incomplete' not in statuses.values(), '仍有未覆盖的已知行动字段')
    write_json_atomic(ROOT / 'event_comparisons.json', {'events': comparisons})
    write_json_atomic(ROOT / 'market_diagnostics.json', {'events': market_checks,
        'note': '原行情不改写。原件拆分比例与因子跳变比较；四位小数舍入只是诊断假设。收益对比未计份额取整，不是策略回测。'})
    pd.DataFrame(windows).to_csv(ROOT / 'split_price_windows.csv', index=False, encoding='utf-8-sig')
    write_json_atomic(ROOT / 'remaining_secondary_schedules.json', {'events': remaining_secondary,
        'symbols': statuses, 'full_corporate_actions_directory_audited': False})
    result = {'verified_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'scope_symbols': 13, 'scope_events': 20, 'primary_pdf_count': 32,
        'official_event_window_indexes_verified': len(indexes), 'reviewed_documents': 22, 'reviewed_complete_pages': 37,
        'split_schedules_verified': len(market_checks), 'dividend_fields_verified': 8,
        'previous_unresolved_inventory_items_reconciled': reconciled,
        'sina_events_with_date_or_ratio_correction': sum(bool(c.get('sina_differences')) for c in comparisons),
        'eastmoney_events_with_ratio_correction': sum(bool(c.get('eastmoney_ratio_differences')) for c in comparisons),
        'factor_diagnostics_outside_rounding_envelope': sum(d['outside_rounding_envelope'] for d in market_checks),
        'largest_split_day_return_difference_percentage_points': max(abs(d['return_difference_percentage_points']) for d in market_checks),
        'remaining_secondary_schedules': len(remaining_secondary), 'pool_known_field_status_counts': dict(Counter(statuses.values())),
        'prior_frozen_inputs_verified': len(prior['input_hashes']), 'readiness_inputs_verified': len(readiness['input_hashes']),
        'input_hashes': inputs, 'artifact_hashes': {},
        'known_event_field_evidence_acceptance': True, 'market_series_acceptance': False,
        'execution_integration_acceptance': False, 'full_corporate_actions_directory_audited': False,
        'investment_admission': False, 'full_goal_achieved': False}
    for name in ('primary_events.json', 'event_comparisons.json', 'market_diagnostics.json',
                 'split_price_windows.csv', 'remaining_secondary_schedules.json'):
        file = ROOT / name
        result['artifact_hashes'][file.relative_to(BASE).as_posix()] = digest(file)
    write_json_atomic(ROOT / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'artifact_hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    verify()
