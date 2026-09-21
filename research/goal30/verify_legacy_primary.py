"""核对新增早期原件；保留既有证据快照，不把已知事件核验当作完整目录。"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from research.goal30.verify_dividend_notices import BASE, EVIDENCE, FROZEN, digest, parse_notice, require

LEGACY = EVIDENCE / 'legacy_primary_20260913'
RECENT = EVIDENCE / 'issuer_dividends_510300'
GOLD = EVIDENCE / 'issuer_reports_518880'
OUT = BASE / 'output/goal30/verification/legacy_primary'


def compact(text):
    return re.sub(r'\s+', '', text)


def parse_gold_2018(pages):
    """物理页已目视；严格定位两列表格的份额列，使用Decimal核对收支。"""
    require(len(pages) == 59, '2018年报页数不符')
    identity, distribution, policy = (compact(pages[n - 1]) for n in (5, 9, 28))
    for value in ('2018年年度报告', '华安易富黄金交易型开放式证券投资基金',
                  '基金主代码518880', '交易代码518880', '华安基金管理有限公司',
                  '上市日期2013年7月29日'):
        require(value in identity, f'黄金ETF身份字段不符：{value}')
    quote = '3.3过去三年基金的利润分配情况本基金过去三年没有利润分配。'
    require(quote in distribution, '缺少2016—2018无分红明确表述')
    require('7.4.4.11基金的收益分配政策' in policy, '缺少分配政策章节')
    require('本基金收益分配采用现金方式' in policy and '本基金每年收益分配最多4次' in policy,
            '现金分配政策不符；不能推断合同禁止分红')
    require('7.4.7.9实收基金' in compact(pages[33]), '实收基金表标题缺失')
    table = pages[34].split('7.4.7.10')[0]
    require('2018年1月1日至2018年12月31日' in compact(table), '份额期间不符')
    require('基金份额（份）账面金额' in compact(table), '份额及金额列标题缺失或顺序不符')
    number = r'(-?\d[\d,]*\.\d{2})'
    values = {}
    for key, label in [('begin_shares', '上年度末'), ('subscriptions', '本期申购'),
                       ('redemptions', '本期赎回（以“-”号填列）'), ('end_shares', '本期末')]:
        matches = re.findall(re.escape(label) + r'\s+' + number + r'\s+' + number, table)
        require(len(matches) == 1, f'份额字段缺失或重复：{label}')
        values[key] = Decimal(matches[0][0].replace(',', ''))
    require(values['begin_shares'] > 0 and values['end_shares'] > 0 and values['subscriptions'] >= 0
            and values['redemptions'] <= 0, '份额方向不符')
    residual = values['end_shares'] - values['begin_shares'] - values['subscriptions'] - values['redemptions']
    require(residual == 0, '份额存在未解释净变动')
    require('报告期末基金份额总额' + f"{values['end_shares']:,.2f}" + '份' in identity,
            '实收基金与基本情况的期末份额不符')
    require('本期已分配利润---' in compact(pages[34]), '当期已分配利润并非三列空值')
    return {'symbol': 'SH518880', 'listing_date': '2013-07-29',
            'no_dividend_evidence': {'start': '2016-01-01', 'end': '2018-12-31', 'pages': [9], 'quote': quote},
            'cash_distribution_permitted': True,
            'share_movement_evidence': {'start': '2018-01-01', 'end': '2018-12-31', 'pages': [34, 35],
                **{key: str(value) for key, value in values.items()}, 'unexplained_net_share_change': str(residual)},
            'scope': '份额净额核对仅覆盖2018年度，不证明其他年度或完整事件目录无拆分。'}


def merge_coverage(intervals):
    merged = []
    for start, end in sorted((date.fromisoformat(a), date.fromisoformat(b)) for a, b in intervals):
        require(start <= end, '覆盖期间逆序')
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[a.isoformat(), b.isoformat()] for a, b in merged]


def verify():
    from pypdf import PdfReader

    inputs, pdf_lengths = {}, {}

    def checked(path, expected=None):
        actual = digest(path)
        require(expected is None or actual == expected, f'输入哈希不符：{path}')
        inputs[path.relative_to(BASE).as_posix()] = actual
        return path

    def read_json(path):
        return json.loads(checked(path).read_text(encoding='utf-8'))

    def pdf_pages(directory, source):
        require(source['status'] == 200 and source['url'].startswith('https://www.sse.com.cn/'),
                '不是上交所成功归档的原版来源')
        path = checked(directory / source['file'], source['sha256'])
        require(path.stat().st_size == source['bytes'], '原件文件长度不符')
        if source.get('index_file'):
            checked(directory / source['index_file'])
        pages = [page.extract_text() for page in PdfReader(path).pages]
        pdf_lengths[path] = len(pages)
        return path, pages

    def check_review(directory, source, pages, visual, expected_pages):
        row = visual['files'][source['file']]
        require(row['sha256'] == source['sha256'], '目视核对的PDF版本不一致')
        require(row['pages_reviewed'] == expected_pages, '目视覆盖范围不符')
        if directory == LEGACY:
            require(row['page_count'] == len(pages), '目视记录页数不符')
            require([image['page'] for image in row['images']] == expected_pages, '图片覆盖页缺漏')
            for image in row['images']:
                checked(directory / image['file'], image['sha256'])

    frozen = read_json(FROZEN)
    for path, sha in frozen['input_hashes'].items():
        checked(BASE / path, sha)
    expected_list = [x for x in frozen['actions'] if x['symbol'] == 'SH510300' and x['type'] == 'cash_dividend']
    expected = {x['ex_date']: x for x in expected_list}
    require(len(expected) == len(expected_list) == 10, '冻结已知分红缺漏或重复')
    old = read_json(RECENT / 'verified_events.json')
    old_events = {x['ex_date']: x for x in old['events']}
    legacy_sources, recent_sources = read_json(LEGACY / 'sources.json'), read_json(RECENT / 'sources.json')
    legacy_visual, recent_visual = read_json(LEGACY / 'visual_review.json'), read_json(RECENT / 'visual_review.json')
    events = []
    for directory, sources, visual in [(LEGACY, legacy_sources, legacy_visual), (RECENT, recent_sources, recent_visual)]:
        for source in sources:
            if not source['file'].startswith('510300'):
                continue
            path, pages = pdf_pages(directory, source)
            check_review(directory, source, pages, visual, list(range(1, len(pages) + 1)))
            event, excerpts = parse_notice('\n'.join(pages))
            require(event['ex_date'] in expected and source['announcement_date'] < event['record_date'], '事件或公告日期不符')
            for comparison in (expected[event['ex_date']], old_events[event['ex_date']]):
                for key in ('record_date', 'ex_date', 'payment_date'):
                    require(event[key] == comparison[key], f'原件与既有记录不符：{path.name}/{key}')
                # 冻结浮点金额含0.05500000000000001，容忍仅限表示误差。
                require(abs(Decimal(str(event['cash_per_share'])) - Decimal(str(comparison['cash_per_share'])))
                        < Decimal('1e-12'), f'原件与既有分红金额不符：{path.name}')
            events.append({**event, 'announcement_date': source['announcement_date'],
                           'source_kind': 'issuer_pdf_hosted_by_sse', 'source_url': source['url'],
                           'file': path.relative_to(BASE).as_posix(), 'sha256': source['sha256'],
                           'pages_reviewed': list(range(1, len(pages) + 1)), 'excerpts': excerpts})
    require(len(events) == len({x['ex_date'] for x in events}) == len(expected), '原版事件缺失或重复')
    require({x['ex_date'] for x in events} == set(expected), '原版与冻结已知事件集不一致')

    gold_sources = [x for x in legacy_sources if x['symbol'] == 'SH518880']
    require(len(gold_sources) == 1, '新增黄金报告来源不唯一')
    source = gold_sources[0]
    path, pages = pdf_pages(LEGACY, source)
    check_review(LEGACY, source, pages, legacy_visual, [5, 9, 28, 34, 35])
    gold = {**parse_gold_2018(pages), 'file': path.relative_to(BASE).as_posix(), 'sha256': source['sha256']}
    prior = read_json(GOLD / 'verified_coverage.json')
    intervals = [[gold['no_dividend_evidence']['start'], gold['no_dividend_evidence']['end']]]
    reused = []
    for report in prior['reports']:
        path, pages = pdf_pages(GOLD, report)
        require(report['page_count'] == len(pages), '既有黄金报告页数不符')
        evidence = report['no_dividend_evidence']
        require(set(evidence['pages']).issubset(report['pages_visually_reviewed']), '既有无分红文字页未目视')
        excerpt = compact(''.join(pages[n - 1] for n in evidence['pages']))
        require(evidence['quote'] in excerpt, '既有无分红原文不符')
        end = date.fromisoformat(evidence['end'])
        start = date.fromisoformat(evidence['start'])
        if 'annual' in report['file']:
            require(f'{end.year}年年度报告' in compact(pages[0]), '年报年份不符')
            require(start == date(end.year - 2, 1, 1) and end == date(end.year, 12, 31), '过去三年区间错误')
        else:
            require(f'{end.year}年中期报告' in compact(pages[0]), '中报年份不符')
            require(start == date(end.year, 1, 1) and end == date(end.year, 6, 30), '中报覆盖区间错误')
        intervals.append([evidence['start'], evidence['end']])
        reused.append({'file': path.relative_to(BASE).as_posix(), 'sha256': report['sha256'], **evidence})
    coverage = merge_coverage(intervals)
    require(coverage == [['2016-01-01', '2026-06-30']], '黄金无分红证据不连续或区间改变')
    for path in (Path(__file__).resolve(), Path(__file__).with_name('verify_dividend_notices.py')):
        checked(path)
    return {'verified_at': datetime.now(timezone.utc).isoformat(),
            'previous_goal_turn': 'progress: 内联收益图修复，断网和交互验收通过；本次继续补证。',
            'window': [frozen['protocol']['start'], frozen['protocol']['end']],
            'known_dividend_events': sorted(events, key=lambda e: e['ex_date']),
            'primary_pdf_events': len(events), 'new_primary_pdf_events': 5,
            'all_known_dividend_fields_match': True, 'all_ten_original_issuer_documents_obtained': True,
            'gold_2018_report': gold, 'reused_gold_no_dividend_reports': reused,
            'gold_no_dividend_primary_coverage': coverage,
            'gold_remaining_dividend_gap': ['2026-07-01', frozen['protocol']['end']],
            'new_pdf_count': sum(path.parent == LEGACY for path in pdf_lengths),
            'new_pdf_total_pages': sum(count for path, count in pdf_lengths.items() if path.parent == LEGACY),
            'new_pages_visually_reviewed': sum(len(row['pages_reviewed']) for row in legacy_visual['files'].values()),
            'full_corporate_actions_audit_complete': False, 'lifecycle_pool_audit_complete': False,
            'investment_admission': False, 'full_goal_achieved': False,
            'limitations': ['只核对已知10次分红，按已知公告日查询不证明全历史公告目录完整。',
                            '黄金ETF仍缺2026年7月1日至回放末日的分红完整性证据及部分年度份额资料。',
                            '黄金2018年度份额净额为零，不证明所有拆分、合并事件不存在。',
                            '沿用冻结收益，不重跑或优化参数；现有候选仍未满足逐日平均回撤和原20%最大回撤门槛。'],
            'input_hashes': inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        path = args.output.resolve()
        require(path.is_relative_to(OUT) and path.suffix == '.json', '结果只可新增在legacy_primary核验目录')
        require(path.relative_to(BASE).as_posix() not in result['input_hashes'], '不得覆盖输入')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in
          ('known_dividend_events', 'input_hashes', 'gold_2018_report', 'reused_gold_no_dividend_reports')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
