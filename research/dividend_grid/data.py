"""新研究只读导入与补充证据；行情缺口估值不伪造成交。"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from research.dividend_trend.run import ROOT, digest, write_json
from research.valuation_bands.data import build_calendar
from research.goal30.bond_dividend_evidence import parse_bond_notice

DB = ROOT/'data/formal/market.duckdb'
INDEX = ROOT/'output/dividend_trend/runs/20261110-090752-975367e5/index_gold_prices.csv'
CSI = ROOT/'output/valuation_bands/comparison_100k_20261109/official_H00300_daily.csv'
ETF = ROOT/'data/valuation_bands/snapshots/etfrun-equal-20261109-v1/etf_bars.csv'
EVIDENCE = ROOT/'output/goal30/data_evidence/broad_readiness_20260914'


def frame_read(path, date='date'):
    frame=pd.read_csv(path,parse_dates=[date]).set_index(date).sort_index()
    frame.index=pd.DatetimeIndex(frame.index).as_unit('ns')
    return frame


def download(session,url,path,params=None):
    meta=path.with_suffix(path.suffix+'.metadata.json')
    if not path.exists():
        response=session.get(url,params=params,timeout=(8,25))
        response.raise_for_status()
        path.write_bytes(response.content)
        write_json(meta,{'url':response.url,'sha256':digest(path),
                        'fetched_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat()})
    assert json.loads(meta.read_text(encoding='utf-8'))['sha256']==digest(path)
    return path.read_bytes()


def prepare(directory):
    directory=directory.resolve()
    if (directory/'manifest.json').exists():
        raise ValueError('已完成数据目录不覆盖')
    directory.mkdir(parents=True,exist_ok=True)
    raw=directory/'raw';raw.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    calendar=build_calendar('2014-01-02','2026-11-04')
    red=frame_read(INDEX).red.reindex(calendar)
    assert red.notna().all() and red.gt(0).all()
    csi=frame_read(CSI).close.reindex(calendar)
    assert csi.loc['2016-09-12':].notna().all()
    warmraw=download(session,'https://www.csindex.com.cn/csindex-home/perf/index-perf',
        raw/'H00300_warmup_2016.json',{'indexCode':'H00300','startDate':'20160101','endDate':'20160911'})
    warmbody=json.loads(warmraw)
    assert str(warmbody['code'])=='200' and warmbody['data']
    warm=pd.DataFrame(warmbody['data'])
    assert warm.indexCode.eq('H00300').all() and warm.indexNameCnAll.eq('沪深300全收益指数').all()
    warm['date']=pd.to_datetime(warm.tradeDate,format='%Y%m%d')
    warmseries=warm.set_index('date').close.astype(float)
    csi=csi.combine_first(warmseries.reindex(calendar))
    assert csi.loc['2016-01-04':].notna().all(), '相对收益预热禁止填未来值'
    with duckdb.connect(str(DB),read_only=True) as con:
        bond=con.execute("SELECT b.date,b.open,b.close,b.volume,b.amount,b.source,a.adj_factor "
            "FROM bars_raw b LEFT JOIN adj_factor a USING(symbol,date) "
            "WHERE b.symbol='SH511260' AND b.date<='2026-11-04' ORDER BY b.date").df().set_index('date')
        gapnav=con.execute("SELECT nav_date,unit_nav,ann_date,source FROM nav WHERE symbol='SH511260' "
                           "AND nav_date='2018-08-17' ORDER BY ann_date").df()
    bond.index=pd.DatetimeIndex(bond.index).as_unit('ns')
    assert not bond.index.has_duplicates and bond.index[0]==pd.Timestamp('2017-08-24')
    missing=calendar[calendar>=bond.index[0]].difference(bond.index)
    assert missing.equals(pd.DatetimeIndex(['2018-08-17']))
    assert len(gapnav)==1 and float(gapnav.unit_nav.iloc[0])>0
    gapnav.to_csv(directory/'bond_gap_nav.csv',index=False)
    qq=download(session,'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get',
        raw/'bond_gap_qq.json',{'param':'sh511260,day,2018-08-14,2018-08-21,30,'})
    qqrows=json.loads(qq)['data']['sh511260']['day']
    assert all(row[0]!='2018-08-17' for row in qqrows)
    for day in ['2018-08-15','2018-08-16','2018-08-20']:
        quoted=next(row for row in qqrows if row[0]==day)
        assert abs(float(quoted[2])-bond.loc[day,'close'])<1e-7
    bond.to_csv(directory/'bond_source.csv',index_label='date')
    bondmark=bond.close.reindex(calendar)
    bondmark.loc['2018-08-17']=float(gapnav.unit_nav.iloc[0])
    # 只填上市前的无持仓记账占位；这些天明确不可交易且目标为零。
    bondmark.loc[calendar<'2017-08-24']=1.
    assert bondmark.notna().all()
    factor=bond.adj_factor.reindex(calendar).ffill().fillna(1.)
    proxy=pd.DataFrame({'red':red,'bond':bondmark*factor,'csi':csi},index=calendar)
    proxy['bond_tradeable']=calendar.isin(bond.index)
    proxy['bond_listed']=calendar>=pd.Timestamp('2017-08-24')
    proxy.to_csv(directory/'proxy.csv',index_label='date')
    etf=frame_read(ETF,'trade_date')
    etf=etf[etf.symbol.eq('SH512890')].loc[:'2026-11-04']
    expected=calendar[calendar>='2019-01-18']
    assert expected.difference(etf.index).equals(pd.DatetimeIndex(['2021-10-22']))
    actual=pd.DataFrame(index=expected)
    for asset,source in [('red',etf),('bond',bond)]:
        actual[asset]=source.close.reindex(expected)
        actual[asset+'_open']=source.open.reindex(expected)
        amount=source.amount_cny if asset=='red' else source.amount
        volume=source.volume_shares if asset=='red' else source.volume*100
        actual[asset+'_amount']=amount.reindex(expected)
        actual[asset+'_volume']=volume.reindex(expected)
        actual[asset+'_adv']=amount.rolling(20,min_periods=20).mean().reindex(expected)
        actual[asset+'_tradeable']=expected.isin(source.index)
    actual.loc['2021-10-22','red']=actual.loc['2021-10-21','red']/2
    assert actual[['red','bond']].notna().all().all()
    actual.to_csv(directory/'etf.csv',index_label='date')
    # 重用四份已逐页核验原件，补充九月场内分红，拒绝直接采用场外日期。
    verified=json.loads((EVIDENCE/'verified_bond_events.json').read_text(encoding='utf-8'))
    events=[x['event'] for x in verified['events']]
    sources=[{'file':str((EVIDENCE/x['source']['file']).relative_to(ROOT)),
              'sha256':x['source']['sha256'],'url':x['source']['url']} for x in verified['events']]
    for item in sources:
        assert digest(ROOT/item['file'])==item['sha256']
    parameters=json.loads((EVIDENCE/'511260_dividend_index_metadata.json').read_text(encoding='utf-8'))['parameters']
    parameters.update(START_DATE='2026-09-01',END_DATE='2026-11-04')
    session.headers.update({'Referer':'https://www.sse.com.cn/','User-Agent':'Mozilla/5.0'})
    index=download(session,'https://query.sse.com.cn/commonQuery.do',raw/'bond_new_dividend_index.json',parameters)
    entries=json.loads(index)['result']
    notices=[x for x in entries if '分红公告' in x.get('TITLE','')]
    assert len(notices)==1,[(x.get('TITLE'),x.get('SSEDATE')) for x in entries]
    notice=notices[0]
    url='https://www.sse.com.cn'+notice['URL']
    pdf=download(session,url,raw/'bond_202609_dividend.pdf')
    from pypdf import PdfReader
    text='\n'.join(page.extract_text() or '' for page in PdfReader(BytesIO(pdf)).pages)
    (raw/'bond_202609_dividend.txt').write_text(text,encoding='utf-8')
    parsed=parse_bond_notice(text)
    events.append(parsed['event'])
    sources.append({'file':str((raw/'bond_202609_dividend.pdf').relative_to(ROOT)),
                    'sha256':digest(raw/'bond_202609_dividend.pdf'),'url':url})
    new_evidence={}
    for asset,symbol in [('bond','SH511260'),('red','SH512890')]:
        content=download(session,f'https://finance.sina.com.cn/realstock/company/{symbol.lower()}/hfq.js',
                         raw/f'{asset}_sina_actions.js')
        from research.goal30.data_evidence import parse_sina
        all_events=parse_sina(content.decode('utf-8'),symbol)
        new_evidence[asset]=all_events
    bonds=[x for x in new_evidence['bond'] if x['type']=='cash_dividend' and x['ex_date']<='2026-11-04']
    assert len(bonds)==len(events)==5
    for item in events:
        match=next(x for x in bonds if x['ex_date']==item['ex_date'])
        assert abs(item['cash_per_share']-match['cash_per_share'])<1e-7
    red_actions=[x for x in new_evidence['red'] if x.get('ex_date',x.get('effective_date',''))<='2026-11-04']
    assert len(red_actions)==1 and red_actions[0]['type']=='split'
    assert red_actions[0]['effective_date']=='2021-10-25' and red_actions[0]['ratio']==2.
    split_source=ROOT/'output/valuation_bands/dividend_comparison_20261109/split_announcement.pdf'
    split_text=''.join(''.join(p.extract_text() or '' for p in PdfReader(split_source).pages).split())
    assert '2021年10月22日（份额拆分日）' in split_text and '拆分比例为1:2' in split_text
    # 新浪因子在恢复交易日才出现；实际拆分登记日以发行人公告为准。
    red_actions[0]={**red_actions[0],'provider_first_quote_date':'2021-10-25','effective_date':'2021-10-22'}
    sources.append({'file':str(split_source.relative_to(ROOT)),'sha256':digest(split_source),
                    'role':'红利拆分日及恢复交易原件'})
    actions={'bond':sorted(events,key=lambda x:x['ex_date']),'red':red_actions,'sources':sources,
             'new_notice_parsed_fields':parsed,'completeness':'5个分红事件与新浪累计分红核对；不等于全市场存续审计'}
    write_json(directory/'actions.json',actions)
    inputs=[DB,INDEX,CSI,ETF,EVIDENCE/'verified_bond_events.json']
    write_json(directory/'manifest.json',{'start':'2016-09-12','end':'2026-11-04',
        'created_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in inputs},
        'files':{str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()},
        'bond_gap':'2018-08-17基金净值仅用于当日估值，缺市场报价且不成交；公布时点未单独证明，另作前价估值压力',
        'bond_gap_nav_available_at':str(gapnav.ann_date.iloc[0]),
        'bond_prelisting_cash':True,'red_etf_listing':'2019-01-18','bond_listing':'2017-08-24',
        'production_changed':False})
    print(json.dumps({'directory':str(directory),'new_notice':parsed,'bond_events':len(events),
        'proxy_rows':len(proxy),'etf_rows':len(actual),'gap_nav':float(gapnav.unit_nav.iloc[0])},ensure_ascii=False,indent=2))


def load(directory):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest['files'].items():
        assert digest(directory/path)==expected,path
    return frame_read(directory/'proxy.csv'),frame_read(directory/'etf.csv'),json.loads((directory/'actions.json').read_text(encoding='utf-8'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
