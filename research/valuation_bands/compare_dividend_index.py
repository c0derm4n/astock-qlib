"""新增官方红利低波全收益指数比较，保留历史比较和正式服务。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .audit import independent_metrics
from .data import build_calendar, sha256_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/valuation_bands/dividend_comparison_510880_20261109/comparison.json'
ENDPOINT = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
FACTSHEET = 'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/H30269factsheet.pdf'
FUND_SUMMARY = 'https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-18/512890_20260318_FNWM.pdf'


def todo_write(output, phase):
    """环境未提供todo工具，以研究状态文件记录阶段与实际完成情况。"""
    phases = ['official_data', 'computation', 'verification']
    current = phases.index(phase) if phase in phases else len(phases)
    tasks = [{'task': name, 'status': 'completed' if i < current else
              ('in_progress' if i == current else 'pending')}
             for i, name in enumerate(phases)]
    write_json(output / 'status.json', {'tasks': tasks, 'phase': phase})


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def fetch_year(job):
    code, year, start, end, output = job
    raw = output / 'raw' / f'{code}_{year}.json'
    metadata = raw.with_suffix('.metadata.json')
    if not raw.exists():
        params = {'indexCode': code,
                  'startDate': max(start, f'{year}-01-01').replace('-', ''),
                  'endDate': min(end, f'{year}-12-31').replace('-', '')}
        session = requests.Session()
        session.trust_env = False
        response = session.get(ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if str(payload.get('code')) != '200' or not payload.get('data'):
            raise ValueError(f'{code}/{year}: 无有效官方数据')
        raw.write_bytes(response.content)
        write_json(metadata, {'url': response.url, 'http_status': response.status_code,
                             'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                             'sha256': sha256_file(raw)})
    if not metadata.exists() or json.loads(metadata.read_text(encoding='utf-8'))['sha256'] != sha256_file(raw):
        raise ValueError(f'原始数据归档哈希不一致: {raw}')
    payload = json.loads(raw.read_text(encoding='utf-8'))
    frame = pd.DataFrame(payload['data'])
    if not frame.indexCode.eq(code).all():
        raise ValueError('官方返回的指数代码不匹配')
    if code == 'H20269' and not frame.indexNameCnAll.eq('中证红利低波动全收益指数').all():
        raise ValueError('不能用价格指数冒充全收益指数')
    frame['date'] = pd.to_datetime(frame.tradeDate, format='%Y%m%d')
    return frame.set_index('date')[['indexCode', 'indexNameCnAll', 'tradeDate', 'close', 'changePct']].sort_index()


def checked_series(frame, calendar):
    if frame.index.has_duplicates:
        raise ValueError('日频数据出现重复日期')
    excluded = frame.loc[~frame.index.isin(calendar), ['tradeDate', 'close']]
    selected = frame.loc[frame.index.isin(calendar)].sort_index()
    if not selected.index.equals(calendar):
        raise ValueError(f'官方日频数据不完整: {calendar.difference(selected.index).tolist()}')
    prices = selected.close.astype(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('指数点位无效')
    return selected, excluded


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw').mkdir(exist_ok=True)
    todo_write(output, 'official_data')
    protected = [BASE, ROOT / 'output/strategies/current.json',
                 ROOT / 'strategy_registry.json', ROOT / 'data/formal/market.duckdb']
    before = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected}
    base = json.loads(BASE.read_text(encoding='utf-8'))
    start, end = base['start'], base['end']
    assert start == '2016-09-12' and end == '2026-11-04'
    calendar = build_calendar(start, end)
    jobs = [(code, year, start, end, output) for code in ('H20269', 'H30269')
            for year in range(int(start[:4]), int(end[:4]) + 1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch_year, jobs))
    by_code = {}
    excluded = []
    for code in ('H20269', 'H30269'):
        frame = pd.concat([f for job, f in zip(jobs, fetched) if job[0] == code])
        by_code[code], removed = checked_series(frame, calendar)
        excluded.append(removed.assign(index_code=code))
        by_code[code].to_csv(output / f'official_{code}_daily.csv')
    pd.concat(excluded).to_csv(output / 'excluded_non_sessions.csv')
    for filename, url in [('H30269factsheet.pdf', FACTSHEET), ('512890_fund_summary.pdf', FUND_SUMMARY)]:
        evidence = output / 'raw' / filename
        if not evidence.exists():
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, timeout=30)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError(f'官方资料不是PDF: {filename}')
            evidence.write_bytes(response.content)
    todo_write(output, 'computation')
    initial = 1_000_000.
    close = by_code['H20269'].close.astype(float)
    nav = initial * close / close.iloc[0]
    metrics = independent_metrics(nav)
    annual = metrics.pop('annual')
    row = {'strategy_id': 'dividend_lowvol_H20269_total_return',
           'label': '中证红利低波动全收益指数H20269',
           'status': 'official_total_return_index_recomputed',
           'account_basis': '100%全收益指数持有基准，指数内部股息再投；未扣基金和交易费用',
           'invested_during_2018': True, **metrics}
    daily = pd.DataFrame({'total_return_close': close,
                          'price_index_close': by_code['H30269'].close,
                          'nav': nav, 'drawdown': 1 - nav / nav.cummax()})
    daily.to_csv(output / 'daily_index_nav.csv')
    pd.DataFrame(annual).to_csv(output / 'annual_returns_H20269.csv', index=False)
    protocol = {'start': start, 'end': end, 'full_years': list(range(2017, 2026)),
                'initial_capital': initial, 'price_index': 'H30269', 'total_return_index': 'H20269',
                'index_publication_date': '2013-12-19',
                'initialization': '起始交易日收盘按指数点位建立100%持有基准，非ETF订单模拟',
                'dividends': 'H20269已包含股息再投资，不重复加息，不拼接512890',
                'costs': '未扣基金管理/托管费、跟踪误差、佣金、滑点及复制交易成本',
                'cagr_basis': 'actual_calendar_days / 365.2425', 'drawdown_frequency': 'daily_close',
                'calendar': 'XSHG完整交易日；剔除接口返回的非交易日占位，不补值',
                'endpoint': ENDPOINT, 'factsheet_url': FACTSHEET, 'fund_summary_url': FUND_SUMMARY,
                'data_files': {p.name: sha256_file(p) for p in sorted((output / 'raw').glob('*'))},
                'protected_inputs': before, 'code_sha256': sha256_file(Path(__file__))}
    write_json(output / 'protocol.json', protocol)
    result = {**base, 'comparison_scope': '新增2016起官方全收益指数；512890上市前现金行仅供参考，不作2018抗跌排名',
              'rows': base['rows'] + [row],
              'index_extension': {'protocol': 'protocol.json', 'raw_data_source': 'csindex_official',
                                  'code_sha256': protocol['code_sha256']},
              'notes': base['notes'] + ['H20269为含股息再投的指数理论基准，覆盖2018年，未扣基金费用和复制成本；不是512890上市前收益。']}
    write_json(output / 'comparison.json', result)
    pd.DataFrame(result['rows']).to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    todo_write(output, 'verification')
    # 独立收益连乘检查归一化路径，另以年末点位比值核验完整年度。
    alternative = initial * close.pct_change(fill_method=None).fillna(0).add(1).cumprod()
    path_error = float((alternative - nav).abs().max())
    assert path_error < 1e-6, path_error
    year_end = close.groupby(close.index.year).last()
    full_annual = year_end.pct_change(fill_method=None).loc[2017:2025]
    assert int(full_annual.idxmin()) == row['worst_year']
    assert abs(float(full_annual.min()) - row['worst_full_year']) < 1e-12
    peak = initial
    maximum_loss = 0.
    for value in alternative:
        peak = max(peak, value)
        maximum_loss = max(maximum_loss, (peak - value) / peak)
    assert abs(maximum_loss - row['max_drawdown']) < 1e-12
    years = (calendar[-1] - calendar[0]).days / 365.2425
    cagr_check = float(np.expm1(np.log(close.iloc[-1] / close.iloc[0]) / years))
    assert abs(cagr_check - row['cagr']) < 1e-12
    tr_returns = close.pct_change(fill_method=None)
    price_returns = by_code['H30269'].close.astype(float).pct_change(fill_method=None)
    # 股息再投应不低于价格收益；允许两位小数指数点位的舍入误差。
    min_excess = float((tr_returns - price_returns).min())
    assert min_excess > -1e-5, min_excess
    after = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected}
    assert before == after, '旧比较或正式状态被改写'
    assert result['rows'][:-1] == base['rows']
    trough = daily.drawdown.idxmax()
    verification = {'status': 'passed', 'observations': len(daily),
                    'complete_years': len(full_annual), 'independent_nav_max_error': path_error,
                    'independent_annual_and_drawdown_checks': True,
                    'official_index_identity_checked': True,
                    'min_daily_total_minus_price_return': min_excess,
                    'excluded_non_session_rows': sum(len(f) for f in excluded),
                    'drawdown_peak': str(nav.loc[:trough].idxmax().date()),
                    'drawdown_trough': str(trough.date()),
                    'return_2018': float(full_annual.loc[2018]),
                    'six_prior_rows_unchanged': True, 'protected_files_unchanged': before == after,
                    'production_and_frontend_modified': False}
    write_json(output / 'verification.json', verification)
    lines = ['# 七项策略与基准比较', '',
             f'区间：{start}—{end}；完整自然年2017—2025；每日收盘净值计算最大回撤。', '',
             '| 策略或基准 | 年化收益 | 最差完整年 | 最大回撤 |',
             '| --- | ---: | ---: | ---: |']
    for r in result['rows']:
        label = r['label']
        if r['strategy_id'] == 'dividend_lowvol_512890_buy_hold':
            label += '，仅参考'
        worst_years = '、'.join(str(y) for y in r.get('worst_years', [r['worst_year']]))
        lines.append(f"| {label} | {r['cagr']:.2%} | {r['worst_full_year']:.2%}（{worst_years}） | {r['max_drawdown']:.2%} |")
    lines += ['', '## 新增基准口径', '',
              '512890主要采用完全复制法跟踪中证红利低波动价格指数H30269，但不保证基金收益等于指数收益。费用、仓位、复制交易及市场折溢价均会导致偏差。',
              '本次新增的是对应全收益指数H20269，按100%持有、股息再投资计算，不扣基金费用和复制成本；不是ETF上市前的可交易收益。',
              '指数发布日2013-12-19早于本次起点。中证官网按年度下载两条指数日频数据，保留原始响应；剔除非交易日占位，不前填、后填或用月线代替日线。',
              f"2018年全收益为{verification['return_2018']:.2%}；全期最大回撤峰日{verification['drawdown_peak']}，谷日{verification['drawdown_trough']}。", '',
              '## 可比性与限制', '',
              '512890上市前现金行保留为历史参考，其2017/2018零收益不代表红利低波经历熊市不亏；不能参加2018抗跌能力排名。',
              '其余六行覆盖2016起实际策略或指数历史。共同日期仅统一观察区间：PE/PB有仓位上限，其他账户、分红及费用口径仍有差异，不是同风险同费用控制实验。',
              '旧正式策略指标和等权估值研究指标沿用原归档：等权估值非整体法，PB财务MRQ未经验证，历史重建非PIT。本次不重跑旧策略、不切换生产策略。', '',
              '## 复现与验证', '',
              f'运行：`.venv/Scripts/python.exe -m research.valuation_bands.compare_dividend_index --output {output.relative_to(ROOT).as_posix()}`。',
              '输入、协议、原始响应及SHA-256见protocol.json与raw/；规范化日线、日净值、年度收益、comparison.json/csv和verification.json均在本目录。',
              '验证完整交易日覆盖、指数身份、两条独立净值路径、年末年度收益、逐日回撤、价格/全收益关系及旧输入和正式状态哈希。未运行全项目测试。', '',
              f'[中证官方指数资料]({FACTSHEET})；[512890基金产品资料概要]({FUND_SUMMARY})。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    todo_write(output, 'completed')
    print(json.dumps({'new_row': row, 'verification': verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
