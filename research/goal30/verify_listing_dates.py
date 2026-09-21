"""重提取30只ETF原版年报的上市字段，并检查冻结行情、资格和现金回放。

只证明指定池的上市日期及其使用边界，不证明完整上市/清盘或公司行动目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parents[2]
EVIDENCE = BASE / 'output/goal30/data_evidence/lifecycle_2025_reports'
OUTPUT = BASE / 'output/goal30/verification/listing_dates'
SELECTED = BASE / 'research/ex_nasdaq/selected_strategy.json'
CASH = BASE / 'output/goal30/cash_replay/20260913-152542-31f263cf'
EXCHANGES = {'SH': '上海证券交易所', 'SZ': '深圳证券交易所'}
DATE_PATTERN = r'(\d{4})(?:年|-)(\d{1,2})(?:月|-)(\d{1,2})日?'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources(sources, symbols):
    require(len(symbols) == len(set(symbols)) == 30 and 'SH513100' not in symbols,
            '冻结候选池必须恰好为指定30只且不含513100')
    require(len(sources) == len({s['symbol'] for s in sources}) == 30,
            '原版报告缺失或标的重复')
    require({s['symbol'] for s in sources} == set(symbols), '原版报告与冻结30只池不一致')
    require(len({s['file'] for s in sources}) == 30, '原版报告文件重复')
    require(len({s['sha256'] for s in sources}) == 30, '不同标的重复使用同一原版报告')
    for source in sources:
        symbol = source['symbol']
        require(source['status'] == 200, f'{symbol}原版来源未取得成功响应')
        require(source['file'] == f'{symbol[2:]}_2025_annual.pdf', '报告文件名不匹配')
        url = urlparse(source['url'])
        expected_host = 'www.sse.com.cn' if symbol[:2] == 'SH' else 'static.cninfo.com.cn'
        require(url.scheme == 'https' and url.hostname == expected_host,
                f'{symbol}不是本阶段核验的官方来源')
        require(re.search(r'2025年?年度报告', source['title']) is not None
                and '提示' not in source['title'] and '联接' not in source['title'],
                f'{symbol}索引不是ETF年度报告正文')


def parse_listing(text, symbol):
    compact = re.sub(r'\s+', '', text)
    codes = re.findall(r'基金主代码(\d{6})(?!\d)', compact)
    require(codes == [symbol[2:]], f'{symbol}基金主代码缺失、重复或不匹配')
    exchanges = re.findall(r'基金份额上市的证券交易所(上海证券交易所|深圳证券交易所)', compact)
    require(exchanges == [EXCHANGES[symbol[:2]]], f'{symbol}上市交易所不匹配')
    matches = list(re.finditer('上市日期' + DATE_PATTERN, compact))
    require(len(matches) == 1, f'{symbol}上市日期缺失或不唯一')
    listing_date = date(*(int(part) for part in matches[0].groups())).isoformat()
    names = re.findall(r'基金名称(.+?)基金简称', compact)
    require(len(names) == 1 and '交易型开放式' in names[0] and '联接' not in names[0],
            f'{symbol}基金名称缺失或不是ETF正文')
    return {'symbol': symbol, 'issuer_list_date': listing_date, 'fund_name': names[0],
            'exchange': exchanges[0], 'listing_excerpt': matches[0].group()}


def check_market_usage(listings, metadata, bars, eligibility, cases):
    import pandas as pd

    symbols = set(listings)
    selected_meta = metadata[metadata.symbol.isin(symbols)].copy()
    require(len(selected_meta) == len(symbols) and not selected_meta.symbol.duplicated().any(),
            '冻结元数据标的缺失或重复')
    dates = {symbol: pd.Timestamp(item['issuer_list_date']) for symbol, item in listings.items()}
    for row in selected_meta.itertuples(index=False):
        require(pd.Timestamp(row.list_date) == dates[row.symbol], f'{row.symbol}上市日期与冻结元数据不一致')
        require(row.exchange == row.symbol[:2], f'{row.symbol}冻结交易所不一致')
    summaries = {}
    for label, frame in [('bars', bars), ('eligibility', eligibility)]:
        frame = frame[frame.symbol.isin(symbols)].copy()
        require(set(frame.symbol) == symbols, f'{label}未覆盖固定30只')
        require(not frame.duplicated(['date', 'symbol']).any(), f'{label}日期/标的重复')
        sessions = pd.to_datetime(frame.date, errors='raise')
        require(sessions.notna().all(), f'{label}缺少日期')
        prelisting = sessions < frame.symbol.map(dates)
        if label == 'bars':
            require(not prelisting.any(), '冻结行情包含上市前价格')
        else:
            require(frame.eligible.notna().all() and pd.api.types.is_bool_dtype(frame.eligible),
                    '资格字段必须为完整布尔值')
            require(not (prelisting & frame.eligible).any(), '上市前被标记为可买入')
        summaries[label] = {'rows': len(frame), 'symbols': len(symbols),
                            'prelisting_rows': int(prelisting.sum()),
                            'prelisting_violations': 0}
    summaries['cash_cases'] = {}
    for name, case in cases.items():
        trades, holdings = case['trades'], case['holdings']
        require(not trades.empty and not holdings.empty, f'{name}归档为空')
        require(set(trades.side).issubset({'buy', 'sell'}), f'{name}成交方向错误')
        for label, frame in [('trades', trades), ('holdings', holdings)]:
            require(set(frame.symbol).issubset(symbols), f'{name}/{label}包含池外标的')
            sessions = pd.to_datetime(frame.date, errors='raise')
            require(sessions.notna().all() and (sessions >= frame.symbol.map(dates)).all(),
                    f'{name}/{label}出现上市前成交或持仓')
        buys = trades[trades.side.eq('buy')]
        signals = pd.to_datetime(buys.signal_date, errors='raise')
        require(signals.notna().all() and (signals >= buys.symbol.map(dates)).all(),
                f'{name}存在上市前买入信号')
        require((signals < pd.to_datetime(buys.date)).all(), f'{name}买入没有在信号日之后成交')
        summaries['cash_cases'][name] = {'trades': len(trades), 'buys': len(buys),
                                        'holdings': len(holdings), 'prelisting_violations': 0}
    return summaries


def verify():
    import pandas as pd
    from pypdf import PdfReader

    inputs = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(BASE), '输入必须来自本工作区')
        actual = digest(path)
        if expected is not None:
            require(actual == expected, f'输入哈希不一致：{path.name}')
        inputs[path.relative_to(BASE).as_posix()] = actual
        return path

    def read_json(path):
        return json.loads(checked(path).read_text(encoding='utf-8'))

    selected = read_json(SELECTED)
    frozen = read_json(CASH / 'frozen_protocol.json')
    prior_cash = read_json(BASE / 'output/goal30/verification/cash_replay/archive_verification.json')
    for manifest in (selected['hashes'], frozen['input_hashes'], prior_cash['evidence_hashes']):
        for path, expected in manifest.items():
            checked(BASE / path, expected)
    sources = read_json(EVIDENCE / 'sources.json')
    validate_sources(sources['sources'], selected['symbols'])
    require(all(item.get('resolved') for item in sources['failures']), '仍有未解决的报告获取失败')
    visual = read_json(EVIDENCE / 'visual_review.json')
    require(set(visual['files']) == {s['file'] for s in sources['sources']}, '目视记录缺失或多出')
    listings, pages_extracted = {}, 0
    for source in sorted(sources['sources'], key=lambda item: item['symbol']):
        path = checked(EVIDENCE / source['file'], source['sha256'])
        require(path.stat().st_size == source['bytes'] and path.read_bytes().startswith(b'%PDF'),
                '原件大小或PDF标记不一致')
        if source.get('reused_from'):
            checked(BASE / source['reused_from'], source['sha256'])
        reader = PdfReader(path)
        page_texts = [page.extract_text() for page in reader.pages[:15]]
        pages_extracted += len(page_texts)
        require(re.search(r'2025年?年度报告', re.sub(r'\s+', '', page_texts[0])) is not None,
                'PDF首页年度报告标题不匹配')
        matches = [(i + 1, text) for i, text in enumerate(page_texts)
                   if re.search('上市日期' + DATE_PATTERN, re.sub(r'\s+', '', text))]
        require(len(matches) == 1, f"{source['symbol']}上市日期相关页缺失或不唯一")
        page, text = matches[0]
        parsed = parse_listing(text, source['symbol'])
        review = visual['files'][path.name]
        require(review['sha256'] == source['sha256'] and review['pdf_physical_pages'] == [page]
                and review['code_exchange_listing_date_checked'] is True,
                '目视记录的PDF版本、页码或核对状态不匹配')
        image = checked(EVIDENCE / review['rendered_image'], review['rendered_sha256'])
        require(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), '目视图片不是PNG')
        listings[source['symbol']] = {**parsed, 'pdf_physical_page': page, 'page_count': len(reader.pages),
                                      'source_file': path.relative_to(BASE).as_posix(),
                                      'source_url': source['url'], 'sha256': source['sha256']}

    source = BASE / selected['source']
    metadata = pd.read_parquet(checked(source / 'metadata_snapshot.parquet'))
    bars = pd.read_parquet(checked(source / 'market_snapshot.parquet'))
    eligibility = pd.read_parquet(checked(source / 'historical_universe.parquet'))
    cases = {case['name']: {label: pd.read_csv(checked(CASH / f'{label}_{case["name"]}.csv'))
                           for label in ('trades', 'holdings')} for case in frozen['protocol']['cases']}
    require(len(cases) == 3, '现金归档情景缺失或重复')
    usage = check_market_usage(listings, metadata, bars, eligibility, cases)
    for row in listings.values():
        row['matches_frozen_list_date'] = True
    checked(Path(__file__))
    return {'verified_at': datetime.now(timezone.utc).isoformat(), 'listing_dates': listings,
            'primary_reports': len(listings), 'extracted_pages': pages_extracted,
            'visually_reviewed_listing_pages': len(listings), 'all_listing_dates_match': True,
            'market_usage': usage, 'input_hashes': inputs,
            'full_historical_lifecycle_audit_complete': False, 'corporate_actions_audited': False,
            'investment_admission': False, 'full_goal_achieved': False,
            'limitations': ['只重提取每份2025年报前15页，目视核对上市日期所在页，未逐页审计全部年报。',
                           '2025年报确认指定基金上市日期，不证明2026年9月的完整存续/清盘状态。',
                           '未覆盖池外已清盘基金；固定池及策略的历史选择偏差仍存在。',
                           '本核验不改写冻结元数据、策略或投资准入门禁。']}


def save_result(path, result):
    path = Path(path).resolve()
    require(path.is_relative_to(OUTPUT.resolve()) and path.suffix == '.json',
            '输出只能保存为listing_dates核验目录内的JSON')
    require(path not in {(BASE / key).resolve() for key in result['input_hashes']}, '输出不得覆盖输入')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'verified_listing_dates.json')
    args = parser.parse_args()
    result = verify()
    save_result(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('listing_dates', 'input_hashes')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
