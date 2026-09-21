"""从归档公告正文重提取510300分红字段，核对冻结现金回放；不提升全池准入。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
EVIDENCE = BASE / 'output/goal30/data_evidence'
FROZEN = BASE / 'output/goal30/cash_replay/20260913-152542-31f263cf/frozen_protocol.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_notice(text):
    compact = re.sub(r'\s+', '', text)
    require('基金主代码510300' in compact, '公告标的身份不匹配')
    require('华泰柏瑞基金管理有限公司' in compact, '公告管理人不匹配')
    out = {'symbol': 'SH510300', 'type': 'cash_dividend'}
    excerpts = {}
    for key, label in [('record_date', '权益登记日'), ('ex_date', '除息日'),
                       ('payment_date', '现金红利发放日')]:
        matches = list(re.finditer(label + r'(\d{4})年(\d{1,2})月(\d{1,2})日', compact))
        require(len(matches) == 1, f'{label}缺失或不唯一')
        match = matches[0]
        out[key] = date(*(int(value) for value in match.groups())).isoformat()
        excerpts[key] = match.group()
    matches = list(re.finditer(r'本次分红方案[（(]单位[：:]元/10份基金份额[）)]([0-9]+\.[0-9]+)', compact))
    require(len(matches) == 1, '每10份分红字段缺失或不唯一')
    amount = Decimal(matches[0][1])
    require(amount > 0, '现金分红必须为正')
    out['cash_per_share'] = float(amount / 10)
    excerpts['cash_per_share'] = matches[0].group()
    require(out['record_date'] < out['ex_date'] <= out['payment_date'], '公司行动日期顺序错误')
    require('投资者可在现金红利发放日领取现金红利' in compact, '缺少投资者到账语义')
    return out, excerpts


def verify():
    # pypdf来自已配置的文档运行时；不改变项目生产依赖。
    from pypdf import PdfReader

    directory = EVIDENCE / 'issuer_dividends_510300'
    discovery = EVIDENCE / 'issuer_notice_discovery/510300'
    inputs = {}

    def checked(path, expected=None):
        actual = digest(path)
        if expected is not None:
            require(actual == expected, f'原响应哈希不一致：{path.name}')
        inputs[path.relative_to(BASE).as_posix()] = actual
        return path

    def read_json(path):
        return json.loads(checked(path).read_text(encoding='utf-8'))

    frozen = read_json(FROZEN)
    actions = [event for event in frozen['actions'] if event['symbol'] == 'SH510300'
               and event['type'] == 'cash_dividend']
    expected = {event['ex_date']: event for event in actions}
    require(len(expected) == len(actions) == 10, '冻结分红事件缺漏或重复')
    sources = read_json(directory / 'sources.json')
    visual = read_json(directory / 'visual_review.json')
    index_meta = read_json(discovery / 'sina_dividend_index_page1_metadata.json')
    index_path = checked(discovery / 'sina_dividend_index_page1.bin', index_meta['sha256'])
    index = json.loads(index_path.read_bytes())['result']['data']
    rows = index['data']
    require(len(rows) == int(index['total_num']) == 10, '公告索引分页未完整取得')
    require(len({row['id'] for row in rows}) == len(rows), '公告索引ID重复')
    rows_by_date = {row['fbsj'][:10]: row for row in rows}
    require(len(rows_by_date) == len(rows), '公告日期重复，需人工处理多公告情况')
    verified = []
    for announcement_date, row in sorted(rows_by_date.items()):
        primary = [source for source in sources if source['announcement_date'] == announcement_date]
        require(len(primary) <= 1, '同日原始PDF来源不唯一')
        if primary:
            source = primary[0]
            path = checked(directory / source['file'], source['sha256'])
            reader = PdfReader(path)
            body = '\n'.join(page.extract_text() for page in reader.pages)
            review = visual['files'][path.name]
            require(review['sha256'] == digest(path), '目视核验的PDF版本不一致')
            require(review['pages_reviewed'] == list(range(1, len(reader.pages) + 1)), '公告未全部目视核对')
            source_kind = 'issuer_pdf_hosted_by_sse'
            pages_reviewed = review['pages_reviewed']
        else:
            name = 'sina_notice_' + announcement_date
            source = read_json(discovery / (name + '_metadata.json'))
            require(source['index_entry'] == row, '转载正文与索引身份不一致')
            path = checked(discovery / (name + '.bin'), source['sha256'])
            # 这些响应是GB18030公告纯文本；严格解码，不丢弃错误字节。
            body = path.read_bytes().decode('gb18030')
            source_kind = 'issuer_notice_transcribed_by_sina'
            pages_reviewed = None
        require(source['status'] == 200, '来源请求未成功')
        event, excerpts = parse_notice(body)
        require(announcement_date < event['record_date'], '公告时间晚于或等于登记日')
        require(event['ex_date'] in expected, '公告事件不在冻结回放中')
        original = expected[event['ex_date']]
        for key in ('record_date', 'ex_date', 'payment_date'):
            require(event[key] == original[key], f'{key}不匹配：{announcement_date}')
        require(abs(event['cash_per_share'] - original['cash_per_share']) < 1e-12,
                f'分红金额不匹配：{announcement_date}')
        verified.append({**event, 'announcement_date': announcement_date,
                         'source_kind': source_kind, 'source_url': source['url'],
                         'source_file': path.relative_to(BASE).as_posix(),
                         'sha256': digest(path), 'pages_reviewed': pages_reviewed,
                         'excerpts': excerpts, 'matches_frozen_event': True})
    require(len({event['ex_date'] for event in verified}) == len(expected), '分红正文覆盖不完整')
    checked(Path(__file__).resolve())
    return {'verified_at': datetime.now(timezone.utc).isoformat(),
            'window_start': frozen['protocol']['start'], 'window_end': frozen['protocol']['end'],
            'symbol': 'SH510300', 'events': verified, 'matched_events': len(verified),
            'primary_pdf_events': sum(event['source_kind'] == 'issuer_pdf_hosted_by_sse' for event in verified),
            'transcribed_notice_events': sum(event['source_kind'] == 'issuer_notice_transcribed_by_sina' for event in verified),
            'all_known_dividend_fields_match': True,
            'all_ten_original_issuer_documents_obtained': False,
            'full_corporate_actions_audit_complete': False,
            'payment_semantics': '使用投资者现金红利发放日；中登或券商提前划款日不作可用现金日。',
            'limitations': '5份早期正文为新浪转载，未取得原版PDF；索引全页不证明历史事项绝无遗漏；不能提升30只全池门禁。',
            'input_hashes': inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        path = args.output.resolve()
        require(path.is_relative_to(EVIDENCE / 'issuer_dividends_510300') and path.suffix == '.json',
                '核验结果只可保存为issuer_dividends_510300目录下的JSON')
        require(path.as_posix() not in {str((BASE / item).resolve().as_posix()) for item in result['input_hashes']},
                '输出不可覆盖输入')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('events', 'input_hashes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
