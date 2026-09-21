"""从已归档发行人PDF重提取本阶段20项行动；不修改冻结行情或执行器。"""
from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.turnover_budget import require

ROOT = BASE / 'output/goal30/data_evidence/broad_actions_20260914'
READINESS = BASE / 'output/goal30/data_evidence/broad_readiness_20260914/readiness.json'
DATE = r'(\d{4}年\d{1,2}月\d{1,2}日)'
NUMBER = r'(\d+(?:\.\d+)?)'
MANAGERS = {
    'SH510500': '南方基金管理股份有限公司', 'SH512010': '易方达基金管理有限公司',
    'SH512100': '南方基金管理股份有限公司', 'SH512170': '华宝基金管理有限公司',
    'SH512200': '南方基金管理股份有限公司', 'SH512690': '鹏华基金管理有限公司',
    'SH512800': '华宝基金管理有限公司', 'SH515070': '华夏基金管理有限公司',
    'SH515220': '国泰基金管理有限公司', 'SZ159901': '易方达基金管理有限公司',
    'SZ159920': '华夏基金管理有限公司', 'SZ159928': '汇添富基金管理股份有限公司',
    'SZ159995': '华夏基金管理有限公司',
}


def compact(text):
    return re.sub(r'\s+', '', text).translate(str.maketrans('（）：，', '():,'))


def iso(value):
    parts = re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日', value)
    require(parts is not None, '日期格式无效')
    return date(*(int(x) for x in parts.groups())).isoformat()


def extract(text, pattern, field, convert=lambda x: x):
    matches = list(re.finditer(pattern, text))
    require(bool(matches), f'原件缺少字段：{field}')
    values = [convert(m[1]) for m in matches]
    require(len(set(values)) == 1, f'原件字段冲突：{field}')
    return values[0], sorted({m.group() for m in matches})


def identity(text, symbol):
    require(symbol in MANAGERS and MANAGERS[symbol] in text, '原件管理人不符')
    require(re.search(r'(?:基金主代码|基金代码:|二级市场交易代码:)' + symbol[2:] + r'(?!\d)', text),
            '原件基金代码不符')


def split_notice(text, symbol, announcement_date, plan_text=None):
    """模板仅限本次有结果公告的拆分；未知或冲突格式直接拒绝。"""
    text = compact(text)
    identity(text, symbol)
    require('份额拆分结果' in text or '份额合并结果' in text, '不是结果公告')
    evidence = {}
    ratios = []
    for pattern in [r'(?:拆分|合并)比例为(?:1:)?' + NUMBER + r'[。,;]',
                    r'按1:' + NUMBER + r'的拆分比例',
                    r'每1份基金份额拆[成为]' + NUMBER + r'份']:
        for match in re.finditer(pattern, text):
            ratios.append((Decimal(match[1]), match.group(), match[1]))
    require(ratios and len({r[0] for r in ratios}) == 1 and ratios[0][0] > 0,
            '拆分比例缺失、非正或原件内部冲突')
    ratio = max((r[2] for r in ratios), key=len)
    evidence['ratio'] = sorted({r[1] for r in ratios})

    def dated(field, pattern, content=text):
        value, excerpts = extract(content, pattern, field, iso)
        evidence[field] = excerpts
        return value

    if symbol == 'SH510500':
        record = book = dated('record_and_book_date', r'份额拆分日\(' + DATE + r'\)日终')
    elif symbol == 'SH512010':
        require(plan_text is not None, '医药ETF结果未列登记日，需实施公告')
        plan = compact(plan_text)
        identity(plan, symbol)
        record = dated('record_date_from_plan', r'权益登记日:' + DATE, plan)
        book = dated('book_date', r'份额拆分日\(' + DATE + r'\)')
    elif symbol == 'SH512100':
        record = dated('record_date', r'对权益登记日\(' + DATE + r'\)登记')
        book = dated('book_date', r'在份额合并日\(' + DATE + r'\)')
    elif symbol in ('SH512170', 'SH512690'):
        record = dated('record_date', r'对' + DATE + r'\(权益登记日\)登记')
        book = dated('book_date', r'已于' + DATE + r'\(份额拆分日\)')
    elif symbol == 'SH512200':
        record = book = dated('record_and_book_date', r'权益登记日\(合并当日,即' + DATE + r'\)')
    elif symbol == 'SZ159901':
        record = book = dated('record_and_book_date', r'权益登记日\(拆分当日,即' + DATE + r'\)')
    elif symbol == 'SZ159928':
        require(plan_text is not None, '消费ETF需实施公告的登记及除权日期')
        plan = compact(plan_text)
        identity(plan, symbol)
        record = dated('record_date_from_plan', r'权益登记日:' + DATE, plan)
        book = dated('book_date', r'已于' + DATE + r'进行了基金份额拆分')
        require('权益登记日日终' in plan, '消费ETF缺少日终拆分语义')
    else:
        record = book = dated('record_and_book_date', r'份额拆分权益登记日\(' + DATE + r'\)日终')

    if symbol in ('SH512170', 'SH512690'):
        available = dated('new_units_available_date', r'本基金将于' + DATE + r'起恢复(?:二级市场)?交易')
        availability_basis = 'secondary_market_resumption'
    elif symbol == 'SH512800':
        available = dated('new_units_available_date', r'自' + DATE + r'起,本基金最小申购')
        require(available == announcement_date and '自公告当日起' in text, '银行ETF新份额查询日不符')
        evidence['query_on_announcement'] = ['自公告当日起', announcement_date]
        availability_basis = 'new_share_query_and_creation_unit_change'
    elif symbol == 'SH515220':
        require('自公告当日起' in text, '煤炭ETF缺少新份额可查询日')
        available = announcement_date
        evidence['new_units_available_date'] = ['自公告当日起', announcement_date]
        availability_basis = 'new_share_query_on_announcement'
    else:
        available = dated('new_units_available_date', r'自' + DATE + r'起(?:基金份额持有人可通过|通过)')
        availability_basis = 'new_share_query_date'
    if symbol == 'SZ159928':
        official_ex_right = dated('ex_right_date_from_plan', r'除权日:' + DATE, plan)
        require(official_ex_right == available, '消费ETF除权及新份额日期不一致')
        availability_basis = 'explicit_ex_right_date'

    if symbol in ('SH510500', 'SH512100', 'SH512200'):
        require('上进位' in text and '小数点后基金份额均进位为1份' in text,
                '非整数折算缺少上进位规则')
        rounding = 'ceiling_to_integer_manager_funded'
        evidence['rounding'] = ['小数点后基金份额均进位为1份']
    elif symbol == 'SZ159995':
        require('截位法保留至整数位,不足1份部分归入基金资产' in text, '芯片ETF截位规则缺失')
        rounding = 'truncate_to_integer_remainder_to_fund'
        evidence['rounding'] = ['截位法保留至整数位,不足1份部分归入基金资产']
    elif symbol == 'SH515070':
        require('按照“精确分配”的原则进行取整' in text, '人工智能ETF精确分配规则缺失')
        rounding = 'registrar_exact_allocation_integer_ratio'
        evidence['rounding'] = ['按照“精确分配”的原则进行取整']
    else:
        require(Decimal(ratio) == Decimal(ratio).to_integral_value(), '未说明取整的非整数拆分')
        rounding = 'integer_ratio_no_fraction_for_integer_holdings'
        evidence['rounding'] = evidence['ratio']
    require(record <= book < available and announcement_date <= available, '拆分日期顺序无效')
    return {'event': {'symbol': symbol, 'type': 'split', 'share_class': 'exchange_traded',
        'record_date': record, 'book_date': book, 'new_units_available_date': available,
        'availability_basis': availability_basis, 'ratio_decimal': ratio, 'rounding': rounding,
        'old_orders_require_recalculation': True}, 'excerpts': evidence}


def dividend_notice(text, symbol):
    text = compact(text)
    identity(text, symbol)
    require('分红公告' in text or '利润分配公告' in text, '不是分红公告')
    evidence = {}
    value, evidence['amount'] = extract(text,
        r'本次分红方案\(单位:(?:人民币)?元/10份基金份额\)' + NUMBER, '每10份分红', Decimal)
    require(value > 0, '现金分红必须为正')
    sections = text.split('与分红相关的其他信息')
    require(len(sections) == 2, '分红日期表缺失或不唯一')
    table = sections[1].split('分红对象')[0]
    fields = [('record_date', '权益登记日'), ('ex_date', '除息日'), ('announced_payment_date', '现金红利发放日')]
    result = {'symbol': symbol, 'type': 'cash_dividend', 'share_class': 'exchange_traded',
              'cash_per_share_decimal': str(value / 10)}
    off_exchange = {}
    for i, (field, label) in enumerate(fields):
        parts = table.split(label)
        require(len(parts) == 2, f'分红表字段不唯一：{label}')
        cell = parts[1].split(fields[i+1][1])[0] if i < 2 else parts[1]
        matches = list(re.finditer(DATE + r'(\(场内\)|\(场外\))?', cell))
        require(matches and ''.join(m.group() for m in matches) == cell, f'日期表含未识别内容：{label}')
        if any(m[2] for m in matches):
            onsite = [m for m in matches if m[2] == '(场内)']
            offshore = [m for m in matches if m[2] == '(场外)']
            require(len(matches) == 2 and len(onsite) == len(offshore) == 1, '场内外日期不唯一')
            chosen = onsite[0]
            off_exchange[field] = iso(offshore[0][1])
        else:
            require(len(matches) == 1 and (symbol != 'SZ159901' or field == 'ex_date'), '份额类别不明确')
            chosen = matches[0]
        result[field] = iso(chosen[1])
        evidence[field] = [cell]

    if symbol == 'SZ159920':
        phrase = '权益登记日申请赎回或卖出的基金份额享有本次分红权益,权益登记日买入的基金份额不享有本次分红权益'
        require(phrase in text and result['record_date'] == result['ex_date'], '恒生ETF同日权益规则缺失或日期冲突')
        result['entitlement_basis'] = 'holdings_before_record_day_trades'
        result['payment_semantics'] = 'custodian_outgoing_transfer_actual_broker_receipt_unconfirmed'
        evidence['entitlement'] = [phrase]
        require('自本基金托管账户划出' in text and '投资者具体收到现金红利的时间遵循' in text, '恒生ETF支付语义缺失')
        evidence['payment_semantics'] = ['自本基金托管账户划出', '投资者具体收到现金红利的时间遵循']
    else:
        require(result['record_date'] < result['ex_date'], '普通场内事件登记须早于除息')
        result['entitlement_basis'] = 'record_date_close_holdings'
        if symbol == 'SH515220':
            require('权益登记日下午上海证券交易所交易结束后' in text and '投资者可在红利发放日领取现金红利' in text,
                    '煤炭ETF场内收盘登记及投资者领取语义缺失')
            result['payment_semantics'] = 'scheduled_investor_claim_date_if_designated_broker'
            evidence['entitlement'] = ['权益登记日下午上海证券交易所交易结束后']
            evidence['payment_semantics'] = ['投资者可在红利发放日领取现金红利']
        elif symbol == 'SZ159901':
            require('场内份额权益登记日下午深圳证券交易所交易结束后' in text and
                    '投资者具体收到现金红利的时间遵循' in text, '深证100场内权益或到账语义缺失')
            result['payment_semantics'] = 'clearing_participant_transfer_actual_broker_receipt_unconfirmed'
            evidence['entitlement'] = ['场内份额权益登记日下午深圳证券交易所交易结束后']
            evidence['payment_semantics'] = ['投资者具体收到现金红利的时间遵循']
        else:
            require(symbol == 'SH512690', '未审核的分红模板')
            phrase = '权益登记日申请申购或买入的基金份额享有本次分红权益'
            require(phrase in text, '酒ETF权益语义缺失')
            result['payment_semantics'] = 'announced_distribution_date_actual_broker_receipt_unconfirmed'
            evidence['entitlement'] = [phrase]
    require(result['ex_date'] <= result['announced_payment_date'], '分红支付日期早于除息')
    result.update(actual_receipt_confirmed=False, actual_cash_available_date=None)
    return {'event': result, 'excerpts': evidence, 'off_exchange_dates': off_exchange}


def parse_sources():
    from pypdf import PdfReader
    sources = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))['sources']
    primary = [s for s in sources if s['source_kind'].startswith('issuer_pdf')]
    parsed = []
    for source in primary:
        if source['kind'] == 'plan':
            continue
        require(digest(ROOT / source['file']) == source['sha256'], '原PDF指纹改变')
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(ROOT / source['file']).pages)
        plans = [s for s in primary if s['event_key'] == source['event_key'] and s['kind'] == 'plan']
        plan_text = None
        if source['symbol'] in ('SH512010', 'SZ159928') and source['kind'] == 'result':
            require(len(plans) == 1, '实施公告缺失或不唯一')
            require(digest(ROOT / plans[0]['file']) == plans[0]['sha256'], '实施公告指纹改变')
            plan_text = '\n'.join(page.extract_text() or '' for page in PdfReader(ROOT / plans[0]['file']).pages)
        if source['kind'] == 'result':
            item = split_notice(text, source['symbol'], source['announcement_date'], plan_text)
        else:
            item = dividend_notice(text, source['symbol'])
        event = item['event']
        event['event_id'] = source['event_key']
        event['announcement_date'] = source['announcement_date']
        if event['type'] == 'cash_dividend':
            require(source['announcement_date'] < event['record_date'], '分红公告晚于权益日')
        item.update(source=source, supporting_plans=plans, evidence_status='known_event_fields_primary_verified',
                    execution_integration_status='not_integrated_or_validated_in_frozen_ledger')
        parsed.append(item)
    return sorted(parsed, key=lambda x: x['event']['event_id'])


if __name__ == '__main__':
    items = parse_sources()
    write_json_atomic(ROOT / 'primary_events.json', {'events': items,
        'scope': '20_known_events_only', 'full_corporate_actions_directory_audited': False,
        'investment_admission': False, 'full_goal_achieved': False})
    print(json.dumps({'primary_events': len(items)}, ensure_ascii=False))
