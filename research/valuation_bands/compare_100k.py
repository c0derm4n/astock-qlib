"""从归档历史生成10万元归一化对照；不重跑订单、不更改正式服务。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research.sp500.benchmark import parse_chart
from .compare_dividend_index import fetch_year, checked_series, write_json
from .data import build_calendar, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/valuation_bands/comparison_100k_20261109'
OLD = ROOT / 'output/valuation_bands/dividend_index_comparison_20261109'
INITIAL = 100_000.
START, END = '2016-09-12', '2026-11-04'


def todo_write(phase):
    phases = ['data', 'calculation', 'visualization', 'verification']
    current = phases.index(phase) if phase in phases else len(phases)
    write_json(OUT / 'status.json', {'phase': phase, 'tasks': [
        {'task': p, 'status': 'completed' if i < current else
         ('in_progress' if i == current else 'pending')} for i, p in enumerate(phases)]})


def yahoo_series(name, symbol):
    path = OUT / 'raw' / f'{name}.json'
    meta = path.with_suffix('.metadata.json')
    if not path.exists():
        params = {'period1': int(pd.Timestamp('2016-09-01', tz='UTC').timestamp()),
                  'period2': int(pd.Timestamp('2026-11-05', tz='UTC').timestamp()),
                  'interval': '1d', 'events': 'div,splits'}
        # 本环境标准网络路由可访问Yahoo公开行情，不使用凭证。
        response = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/' + symbol,
                                params=params, timeout=(8, 30), headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        payload = response.json()['chart']['result'][0]
        assert payload['meta']['symbol'] == symbol
        path.write_bytes(response.content)
        write_json(meta, {'url': response.url, 'sha256': sha256_file(path),
                          'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                          'symbol': symbol, 'currency': payload['meta']['currency'],
                          'timezone': payload['meta']['exchangeTimezoneName']})
    assert json.loads(meta.read_text(encoding='utf-8'))['sha256'] == sha256_file(path)
    chart = json.loads(path.read_text(encoding='utf-8'))['chart']['result'][0]
    frame = parse_chart(chart)
    frame.to_csv(OUT / f'{name}.csv')
    return frame


def read_curve(path, date, value):
    frame = pd.read_csv(path, usecols=[date, value], parse_dates=[date]).set_index(date)[value]
    frame.index = pd.DatetimeIndex(frame.index).as_unit('ns')
    return frame.loc[START:END]


def metrics(nav):
    assert nav.index[0] == pd.Timestamp(START) and nav.index[-1] == pd.Timestamp(END)
    assert not nav.index.has_duplicates and nav.index.is_monotonic_increasing
    assert np.isfinite(nav).all() and (nav > 0).all()
    assert abs(nav.iloc[0] - INITIAL) < 1e-7
    years = (nav.index[-1] - nav.index[0]).days / 365.2425
    annual = []
    for year, group in nav.groupby(nav.index.year):
        prior = nav.loc[nav.index < pd.Timestamp(year, 1, 1)]
        base = prior.iloc[-1] if len(prior) else INITIAL
        annual.append({'year': int(year), 'return': float(group.iloc[-1] / base - 1),
                       'complete': 2017 <= year <= 2025})
    worst = min((a for a in annual if a['complete']), key=lambda a: a['return'])
    return {'start': START, 'end': END, 'initial_capital': INITIAL, 'observations': len(nav),
            'final_nav': float(nav.iloc[-1]), 'cagr': float((nav.iloc[-1] / INITIAL) ** (1 / years) - 1),
            'max_drawdown': float((1 - nav / nav.cummax()).max()),
            'worst_full_year': worst['return'], 'worst_year': worst['year'], 'annual': annual}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'raw').mkdir(exist_ok=True)
    todo_write('data')
    old = json.loads((OLD / 'comparison.json').read_text(encoding='utf-8'))
    protected = [OLD / 'comparison.json', ROOT / 'output/strategies/current.json',
                 ROOT / 'data/formal/market.duckdb', ROOT / 'strategy_registry.json']
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected}
    calendar = build_calendar(START, END)
    jobs = [('H00300', y, START, END, OUT) for y in range(2016, 2027)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(fetch_year, jobs))
    csi, excluded = checked_series(pd.concat(frames), calendar)
    assert csi.indexNameCnAll.eq('沪深300全收益指数').all()
    csi.to_csv(OUT / 'official_H00300_daily.csv')
    excluded.to_csv(OUT / 'excluded_non_sessions_H00300.csv')
    with ThreadPoolExecutor(max_workers=2) as pool:
        sp_future = pool.submit(yahoo_series, 'sp500_total_return', '^SP500TR')
        fx_future = pool.submit(yahoo_series, 'usdcny', 'CNY=X')
        sp, fx = sp_future.result(), fx_future.result()
    import exchange_calendars as xcals
    nyse = pd.DatetimeIndex(xcals.get_calendar('XNYS').sessions_in_range(START, END)).as_unit('ns')
    us = sp.close.loc[START:END]
    assert us.index.equals(nyse), '标普全收益日线缺少美股交易日'
    assert fx.index[0] <= pd.Timestamp(START) and fx.index[-1] >= pd.Timestamp(END)
    # 汇率日与美股日的并集保留美股假期汇率波动，不只在中国开市日抽样算回撤。
    us_calendar = nyse.union(fx.index[(fx.index >= START) & (fx.index <= END)]).sort_values()
    usd = sp.close.reindex(us_calendar, method='ffill', tolerance=pd.Timedelta(days=7))
    rate = fx.close.reindex(us_calendar, method='ffill', tolerance=pd.Timedelta(days=7))
    assert usd.notna().all() and rate.notna().all()
    sp_cny = usd * rate
    # 逐年检查可复算的美元本币收益，并保留该辅助口径。
    usd_normalized = INITIAL * us / us.iloc[0]
    usd_metrics = metrics(usd_normalized)
    usd_annual = {a['year']: a['return'] for a in usd_metrics['annual']}
    for year, expected in {2017: .2183, 2018: -.0438, 2019: .3149,
                           2020: .1840, 2021: .2871, 2022: -.1811,
                           2023: .2629, 2024: .2502}.items():
        assert abs(usd_annual[year] - expected) < .0003, (year, usd_annual[year])
    formal = Path(old['formal_run'])
    research = Path(old['research_run'])
    paths = {
        'A_pe_weekly': (research / 'cases/A_pe_weekly/daily_account.csv', 'session', 'nav'),
        'B_pb_weekly': (research / 'cases/B_pb_weekly/daily_account.csv', 'session', 'nav'),
        'lightgbm': (formal / 'lightgbm_report.csv', 'date', 'account'),
        'trend200_vol15': (formal / 'trend200_vol15_report.csv', 'date', 'account'),
        'dividend_510880_total_return': (ROOT / 'output/valuation_bands/dividend_comparison_510880_20261109/daily_account.csv', 'date', 'nav'),
        'dividend_lowvol_H20269_total_return': (OLD / 'daily_index_nav.csv', 'date', 'nav')}
    labels = {'A_pe_weekly': '沪深300PE（等权）', 'B_pb_weekly': '创业板PB（等权）',
              'lightgbm': 'LightGBM核心卫星', 'trend200_vol15': '通信/黄金',
              'dividend_510880_total_return': '红利ETF510880（再投）',
              'dividend_lowvol_H20269_total_return': '红利低波全收益',
              'csi300_total_return': '沪深300全收益', 'sp500_total_return_cny': '标普500全收益（人民币）'}
    curves, rows, annual_rows, source_hashes = {}, [], [], {}
    todo_write('calculation')
    old_by_id = {r['strategy_id']: r for r in old['rows']}
    for sid, (path, date, value) in paths.items():
        original = read_curve(path, date, value)
        assert original.index.equals(calendar), sid
        source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        curve = INITIAL * original / original.iloc[0]
        curves[sid] = curve
        computed = metrics(curve)
        for field in ['cagr', 'max_drawdown', 'worst_full_year']:
            assert abs(computed[field] - old_by_id[sid][field]) < 1e-12, (sid, field)
        annual_rows += [{'strategy_id': sid, **a} for a in computed.pop('annual')]
        rows.append({**old_by_id[sid], **computed, 'label': labels[sid],
                     'normalization': '归档净值等比例换算至10万元，不是10万元整手成交重跑'})
    for sid, values in [('csi300_total_return', csi.close), ('sp500_total_return_cny', sp_cny)]:
        curves[sid] = INITIAL * values / values.iloc[0]
        computed = metrics(curves[sid])
        annual_rows += [{'strategy_id': sid, **a} for a in computed.pop('annual')]
        rows.append({'strategy_id': sid, 'label': labels[sid], **computed,
                     'account_basis': '100%全收益指数；标普按USD/CNY换算；未扣复制成本、税与汇兑成本',
                     'status': 'historical_index_benchmark', 'investment_reference_only': True})
    pd.DataFrame(rows).to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(annual_rows).to_csv(OUT / 'annual_returns.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'daily_wealth_native_calendars.csv', index_label='date')
    aligned_fx = pd.DataFrame({'sp500_total_return_usd': usd, 'usdcny': rate,
                               'wealth_cny': curves['sp500_total_return_cny']})
    aligned_fx.to_csv(OUT / 'sp500_cny_daily.csv', index_label='date')
    result = {'start': START, 'end': END, 'initial_capital': INITIAL, 'rows': rows,
              'scope': '四条历史策略回放、510880再投代理、三条全收益指数；去掉512890上市前现金行',
              'notes': ['10万元为归一化展示，非按整手/最低佣金重新执行。',
                        '各策略均覆盖2018年；完整自然年2017—2025，2016/2026为部分年。',
                        '标普主行按人民币换算，汇率为USD/CNY；美元本币指标另存。',
                        '跨市场曲线按各市场本地收盘日期作事后对照，并非北京时间同一时点，不用于信号。',
                        '最大回撤使用各序列全部日频数据；不只在中国开市日采样美股。',
                        '全收益指数已含股息再投，不等同可直接交易ETF收益；费用与账户口径差异保留。'],
              'sp500_usd_supplement': usd_metrics}
    write_json(OUT / 'comparison.json', result)
    protocol = {'start': START, 'end': END, 'normalization_capital': INITIAL,
                'full_years': list(range(2017, 2026)), 'protected_hashes': hashes,
                'curve_input_hashes': source_hashes, 'code_sha256': sha256_file(Path(__file__)),
                'raw_data_hashes': {p.name: sha256_file(p) for p in sorted((OUT / 'raw').glob('*'))},
                'sp500_calendar': '全部XNYS交易日与汇率报价日的并集，非交易日报价只向前匹配，容忍7天；不延伸末尾',
                'sp500_timing': '本地日期收盘事后比较；美国11月4日收盘在中国11月5日，非同时估值',
                'normalization': '原策略100万元净值除以首日净值再乘10万元；没有重跑订单和费用',
                'costs': '旧回测保留原费用；指数未扣基金费、复制成本、税及汇兑成本',
                'formal_service_changed': False}
    write_json(OUT / 'protocol.json', protocol)
    # 核验逐日单遍峰值回撤和逐年年末比值，不从图表抽样序列计算风险。
    errors = {}
    for sid, curve in curves.items():
        row = next(r for r in rows if r['strategy_id'] == sid)
        peak, dd = INITIAL, 0.
        for value in curve:
            peak = max(peak, float(value))
            dd = max(dd, (peak - value) / peak)
        assert abs(dd - row['max_drawdown']) < 1e-12
        year_ends = curve.groupby(curve.index.year).last()
        yearly = year_ends.pct_change(fill_method=None).loc[2017:2025]
        assert abs(yearly.min() - row['worst_full_year']) < 1e-12
        alternative = INITIAL * curve.pct_change(fill_method=None).fillna(0).add(1).cumprod()
        errors[sid] = float((alternative - curve).abs().max())
        assert errors[sid] < 1e-6
    assert hashes == {str(p.relative_to(ROOT)): sha256_file(p) for p in protected}
    write_json(OUT / 'verification.json', {'status': 'passed', 'series_count': len(rows),
               'domestic_sessions': len(calendar), 'us_stock_sessions': len(us),
               'sp500_currency_valuation_days': len(sp_cny),
               'six_existing_returns_unchanged': True, 'all_start_at_100000': True,
               'all_cover_2018': True, 'independent_nav_max_errors': errors,
               'annual_sp500_usd_spot_checks': list(range(2017, 2025)),
               'protected_files_unchanged': True, 'production_modified': False})
    # 每周最后一条观察用于可视化，指标仍用每日净值。八条宽表共约530行。
    chart_dates = pd.DatetimeIndex([pd.Timestamp(START)] + list(
        pd.Series(calendar, index=calendar).resample('W-FRI').last().dropna()))
    chart_dates = chart_dates.unique().sort_values()
    wealth, drawdowns = {}, {}
    for sid, curve in curves.items():
        wealth[sid] = curve.reindex(chart_dates, method='ffill', tolerance=pd.Timedelta(days=10)) / 10_000
        daily_dd = curve / curve.cummax() - 1
        drawdowns[sid] = daily_dd.reindex(chart_dates, method='ffill', tolerance=pd.Timedelta(days=10))
    assert all(v.notna().all() for v in wealth.values())
    payload = {'start': START, 'end': END, 'rows': rows, 'series_ids': list(curves),
               'labels': labels, 'dates': [str(d.date()) for d in chart_dates],
               'wealth_wan': {k: [float(x) for x in v] for k, v in wealth.items()},
               'drawdowns': {k: [float(x) for x in v] for k, v in drawdowns.items()},
               'annual': annual_rows}
    write_json(OUT / 'visualization_data.json', payload)
    lines = ['# 10万元起始的八项历史比较', '',
             f'区间{START}—{END}；完整年2017—2025；历史净值归一化至10万元。', '',
             '| 策略或基准 | 期末金额（万元） | 年化收益 | 最差完整年 | 最大回撤 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['label']} | {r['final_nav']/10000:.2f} | {r['cagr']:.2%} | {r['worst_full_year']:.2%}（{r['worst_year']}） | {r['max_drawdown']:.2%} |")
    lines += ['', *result['notes'], '', '净值来源、原始数据哈希和计算方法见protocol.json；逐日净值、年度收益及独立核验见本目录CSV和verification.json。',
              '复跑：`.venv/Scripts/python.exe -X utf8 -m research.valuation_bands.compare_100k`。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    todo_write('visualization')
    print(json.dumps({'rows': rows, 'chart_observations': len(chart_dates),
                      'sp500_usd': usd_metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
