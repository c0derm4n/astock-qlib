"""红利低波估值带研究的数据准备；只新建本研究的独立数据目录，不修改正式库。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from research.dividend_trend.run import ROOT, digest, write_json
from research.valuation_bands.data import build_calendar

DB = ROOT / 'data/formal/market.duckdb'
TR_SOURCE = ROOT / 'output/dividend_trend/runs/20261110-090752-975367e5/index_gold_prices.csv'
PRICE_OFFICIAL = ROOT / 'output/valuation_bands/dividend_index_comparison_20261109/official_H30269_daily.csv'
ETF_SNAPSHOT = ROOT / 'data/valuation_bands/snapshots/etfrun-equal-20261109-v1/etf_bars.csv'
CSI = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
YIELD_URL = 'https://datacenter.eastmoney.com/api/data/get'
YIELD_PARAMS = {'type': 'RPTA_WEB_TREASURYYIELD', 'sty': 'ALL', 'st': 'SOLAR_DATE',
                'sr': '-1', 'ps': '500'}


def frame_read(path, date='date'):
    frame = pd.read_csv(path, parse_dates=[date]).set_index(date).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).as_unit('ns')
    return frame


def download(session, url, path, params=None):
    meta = path.with_suffix(path.suffix + '.metadata.json')
    if not path.exists():
        response = session.get(url, params=params, timeout=(8, 30))
        response.raise_for_status()
        path.write_bytes(response.content)
        write_json(meta, {'url': response.url, 'sha256': digest(path),
                          'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
    stored = json.loads(meta.read_text(encoding='utf-8'))
    assert stored['sha256'] == digest(path), path
    return path.read_bytes()


def fetch_price_warmup(session, raw, year):
    """H30269价格指数预热；与全收益指数使用同一官方接口与同一校验方式。"""
    path = raw / f'H30269_{year}.json'
    body = json.loads(download(session, CSI, path, {
        'indexCode': 'H30269', 'startDate': f'{year}0101',
        'endDate': f'{year}1231' if year < 2016 else '20160911'}))
    assert str(body['code']) == '200' and body['data'], year
    frame = pd.DataFrame(body['data'])
    assert frame.indexCode.eq('H30269').all(), year
    assert frame.indexNameCnAll.eq('中证红利低波动指数').all(), year
    frame['date'] = pd.to_datetime(frame.tradeDate, format='%Y%m%d')
    return frame.set_index('date').close.astype(float)


def fetch_yield10(session, raw):
    """东方财富国债收益率分页原始响应；只取覆盖起点所需页数。"""
    frames, page = [], 1
    while True:
        body = json.loads(download(session, YIELD_URL, raw / f'yield10_page_{page}.json',
                                   {**YIELD_PARAMS, 'p': str(page), 'pageNo': str(page),
                                    'pageNum': str(page)}))
        assert str(body['code']) == '0', body.get('message')
        result = body['result']
        frame = pd.DataFrame(result['data'])
        assert not frame.empty, page
        frames.append(frame)
        oldest = pd.to_datetime(frame.SOLAR_DATE).min()
        if oldest <= pd.Timestamp('2013-12-31') or page >= int(result['pages']):
            break
        page += 1
    frame = pd.concat(frames, ignore_index=True)
    assert frame.SOLAR_DATE.is_unique
    frame['date'] = pd.to_datetime(frame.SOLAR_DATE)
    frame = frame.set_index('date').sort_index()
    value = pd.to_numeric(frame.EMM00166466, errors='coerce')
    assert value.notna().sum() > 2000
    return value.rename('yield10')


def prepare(directory):
    directory = directory.resolve()
    if (directory / 'manifest.json').exists():
        raise ValueError('已完成数据目录不覆盖')
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'raw'
    raw.mkdir(exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    warmup, start, end = '2014-01-02', '2016-09-12', '2026-11-04'
    calendar = build_calendar(warmup, end)

    tr = frame_read(TR_SOURCE).red.reindex(calendar)
    assert tr.notna().all() and tr.gt(0).all()
    official = frame_read(PRICE_OFFICIAL).close
    warm = pd.concat([fetch_price_warmup(session, raw, year) for year in (2014, 2015, 2016)])
    price = pd.concat([warm[~warm.index.isin(official.index)], official]).sort_index()
    price = price.reindex(price.index.intersection(calendar))
    price = price.reindex(calendar)
    assert price.loc[pd.Timestamp(start):].notna().all(), '2016-09-12起价格指数禁止缺口'
    assert price.notna().all()

    y10 = fetch_yield10(session, raw).reindex(calendar).ffill()
    assert y10.notna().all(), '10年期国债收益率缺口'
    assert y10.loc[pd.Timestamp(start):].between(0.5, 6.5).all()

    with duckdb.connect(str(DB), read_only=True) as con:
        bond = con.execute(
            "SELECT b.date,b.close,a.adj_factor,b.source FROM bars_raw b "
            "LEFT JOIN adj_factor a USING(symbol,date) WHERE b.symbol='SH511260' "
            "AND b.date BETWEEN ? AND ? ORDER BY b.date", [warmup, end]).df()
        gap = con.execute(
            "SELECT nav_date,unit_nav,ann_date FROM nav WHERE symbol='SH511260' "
            "AND nav_date='2018-08-17' ORDER BY ann_date").df()
    bond['date'] = pd.to_datetime(bond.date)
    bond = bond.set_index('date')
    bond.index = pd.DatetimeIndex(bond.index).as_unit('ns')
    assert not bond.index.has_duplicates and bond.index[0] == pd.Timestamp('2017-08-24')
    assert calendar[calendar >= bond.index[0]].difference(bond.index).equals(
        pd.DatetimeIndex(['2018-08-17']))
    bond['adj_factor'] = bond.adj_factor.ffill().fillna(1.)
    bond_total = (bond.close * bond.adj_factor).reindex(calendar)
    assert len(gap) == 1
    bond_total.loc['2018-08-17'] = float(gap.unit_nav.iloc[0])
    listed = calendar >= bond.index[0]
    # 上市前价格置为常数1且目标权重为零；该段相当于现金，不伪造上市前收益。
    bond_total.loc[~listed] = 1.
    assert bond_total.notna().all()

    panel = pd.DataFrame({'red': tr, 'price': price.reindex(calendar),
                          'bond': bond_total, 'yield10': y10.reindex(calendar),
                          'bond_listed': listed}, index=calendar)
    panel.index.name = 'date'
    panel.to_csv(directory / 'panel.csv', index_label='date')

    etf = frame_read(ETF_SNAPSHOT, 'trade_date')
    etf = etf[etf.symbol.eq('SH512890')].loc[:end]
    expected = calendar[calendar >= '2019-01-18']
    assert expected.difference(etf.index).equals(pd.DatetimeIndex(['2021-10-22']))
    # 2021-10-22为份额拆分登记日，当日只按前一有效报价估值、不允许成交。
    etf_red = (etf.close * np.where(etf.index >= pd.Timestamp('2021-10-22'), 2., 1.))
    etf_red = etf_red.reindex(expected)
    etf_red.loc['2021-10-22'] = etf_red.loc['2021-10-21']
    etf_panel = pd.DataFrame({
        'red': etf_red,
        'bond': panel.bond.reindex(expected),
        'bond_listed': panel.bond_listed.reindex(expected)}, index=expected)
    assert etf_panel.notna().all().all()
    etf_panel.index.name = 'date'
    etf_panel.to_csv(directory / 'etf_panel.csv', index_label='date')

    write_json(directory / 'manifest.json', {
        'start': start, 'end': end, 'warmup_start': warmup,
        'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                          [TR_SOURCE, PRICE_OFFICIAL, ETF_SNAPSHOT, DB]},
        'files': {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'},
        'bond_gap': '2018-08-17只有基金净值(102.057)可作估值，无市场报价、当日不得成交',
        'bond_prelisting_cash': True, 'bond_listing': '2017-08-24',
        'etf_listing': '2019-01-18', 'etf_split_date': '2021-10-22',
        'yield10_note': '东方财富RPTA_WEB_TREASURYYIELD原始分页JSON留档；按XSHG交易日向前填充',
        'production_changed': False})
    print(json.dumps({'directory': str(directory), 'rows': len(panel),
                      'etf_rows': len(etf_panel),
                      'yield10_first': str(panel.yield10.index[0].date()),
                      'price_first': str(panel.price.index[0].date())},
                     ensure_ascii=False, indent=2))


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['files'].items():
        assert digest(directory / path) == expected, path
    return frame_read(directory / 'panel.csv'), frame_read(directory / 'etf_panel.csv'), manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
