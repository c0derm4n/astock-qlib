"""从冻结账本和原价独立核查归因CSV、历史不足分类及跨期闭合。"""
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.broad_action_evidence import READINESS
from research.goal30.group_return_attribution import OUT, COMPONENTS
from research.goal30.primary_research_series import START, END, corrected_market, read
from research.goal30.run_dividend_payment_replay import GROUP
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import PrimaryMarketDays, REFERENCE


def audit():
    manifest=read(OUT/'completion.json');result=read(OUT/'results.json');reg=read(GROUP/'registration.json')
    for rel,h in {**manifest['input_hashes'],**manifest['artifact_hashes']}.items(): require(digest(BASE/rel)==h,'归因输入或产物改变')
    frames=PrimaryMarketDays(corrected_market());prices=frames.market.wide('adj_close')
    benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True).spy_total_cny
    maximum=0.;days=0
    def same(actual,expected,label,tolerance=1e-7):
        nonlocal maximum
        difference=abs(float(actual)-float(expected));maximum=max(maximum,difference)
        require(np.isfinite(difference) and difference<tolerance,label)
    for key,item in result['cases'].items():
        data=read(GROUP/(key+'.json'));pnl=pd.read_csv(OUT/(key+'_pnl.csv'));bridge=pd.read_csv(OUT/(key+'_cash_bridge.csv'))
        lookup={kind:defaultdict(list) for kind in ('trades','actions','holdings')}
        for kind in lookup:
            for row in data[kind]: lookup[kind][row['date']].append(row)
        require(list(pnl.date)==[r['date'] for r in data['report']],'归因日期不同')
        require(list(bridge.date)==[r['date'] for r in data['signals']],'现金桥日期不同')
        previous_q={};nav=cash=reg['policy']['initial_capital'];claims=value=0.;linked={k:0. for k in COMPONENTS}
        for i,(row,report) in enumerate(zip(pnl.to_dict('records'),data['report'])):
            day=row['date'];context=frames.context(day);after_actions=dict(previous_q)
            accrual=0.
            for action in lookup['actions'][day]:
                if action['type']=='split': after_actions[action['symbol']]=action['quantity_after']
                elif action['type']=='accrue_dividend': accrual+=action['amount']
            price_pnl=sum(q*context['close'][s] for s,q in after_actions.items())-value
            timing=commission=slip=0.
            for trade in lookup['trades'][day]:
                direction=1 if trade['side']=='buy' else -1
                timing+=direction*trade['quantity']*(context['close'][trade['symbol']]-context['open'][trade['symbol']])
                commission+=trade['commission']
                slip+=direction*trade['quantity']*(trade['price']-context['open'][trade['symbol']])
            same(row['market_price_pnl']+row['split_rounding_pnl'],price_pnl,'未复权价格和拆分数量不符')
            for column,actual in [('dividend_accrual_pnl',accrual),('trading_time_pnl',timing),('commission_cost',commission),('slippage_cost',slip)]:
                same(row[column],actual,'独立金额项不同：'+column)
            same(report['account']-nav,price_pnl+accrual+timing-commission-slip,'净值变化不闭合')
            rs=report['account']/nav-1;rb=benchmark.iloc[i]/benchmark.iloc[i-1]-1 if i else 0.
            same(row['strategy_return'],rs,'策略收益不同',1e-12);same(row['benchmark_return'],rb,'基准收益不同',1e-12)
            amounts={'cash_opportunity':-(cash+claims)*rb,'invested_excess':price_pnl+accrual-value*rb,
                     'trading_time':timing,'commission':-commission,'slippage':-slip}
            factor=(math.log1p(rs)-math.log1p(rb))/(rs-rb) if abs(rs-rb)>1e-12 else 1/(1+rb)
            for name,amount in amounts.items():
                same(row[name],amount/nav,'相对收益分量不同',1e-12)
                same(row[name+'_linked_log'],amount/nav*factor,'链接对数贡献不同',1e-12)
                linked[name]+=amount/nav*factor
            nav=report['account'];cash=report['cash'];claims=report['dividend_receivable'];value=report['value']
            previous_q={h['symbol']:h['quantity'] for h in lookup['holdings'][day]};days+=1
        profit=item['profit'];years=(benchmark.index[-1]-benchmark.index[0]).days/365.25
        cg=(nav/reg['policy']['initial_capital'])**(1/years)-1
        cb=(benchmark.iloc[-1]/benchmark.iloc[0])**(1/years)-1
        scale=(cg-cb)/(math.log1p(cg)-math.log1p(cb))
        for name in COMPONENTS:
            same(profit['linked_relative_log_contributions'][name],linked[name],'合计对数贡献不同',1e-11)
            same(profit['cagr_gap_allocation_by_log_link'][name],linked[name]/years*scale,'CAGR差额分配不同',1e-11)
        same(sum(linked.values()),math.log(nav/reg['policy']['initial_capital'])-math.log(benchmark.iloc[-1]/benchmark.iloc[0]),'相对增长不闭合',1e-11)
        names=list(item['cash_bridge']['mean_components'])
        reports={r['date']:r for r in data['report']};month_total=0.
        for row,signal in zip(bridge.to_dict('records'),data['signals']):
            report=reports[row['date']];raw=sum(signal['selection_state']['raw_weights'].values())
            if signal['monthly_review']: month_total=raw
            cash_fraction=report['cash']/(report['cash']+report['value'])
            same(row['liquid_cash_fraction'],cash_fraction,'现金桥分母不同',1e-11)
            same(sum(row[n] for n in names),cash_fraction,'现金桥不能相加',1e-11)
            same(row['unfilled_slots_at_monthly_review'],reg['policy']['max_gross_weight']-month_total,'初始空槽比例不同',1e-11)
            same(row['removed_targets_since_monthly_review'],month_total-raw,'月内移除比例不同',1e-11)
            same(row['target_to_current_holdings_gap'],sum(signal['target_weights'].values())-(1-cash_fraction),'目标与持仓差不同',1e-11)
        for name in names: same(item['cash_bridge']['mean_components'][name],bridge[name].mean(),'现金桥均值不同',1e-11)
    history=pd.read_csv(OUT/'history_diagnosis.csv');lookup=history.set_index(['date','symbol']);listing=read(READINESS)['symbols']
    counts=Counter();eligible_count=0;missing=[]
    index=prices.index
    for symbol in prices:
        values=prices[symbol].to_numpy();first=int(np.flatnonzero(np.isfinite(values)&(values>0))[0]);listed=pd.Timestamp(listing[symbol]['listing_date'])
        origin=max(first,int(index.searchsorted(listed)))
        windows=[(pd.Timestamp(e['record_date']),pd.Timestamp(e['effective_date'])) for e in reg['actions'] if e['symbol']==symbol and e['type']=='split']
        def overlap(day): return any(a<=day<b for a,b in windows)
        for i,day in enumerate(index):
            if day<pd.Timestamp(START) or day>pd.Timestamp(END): continue
            if day>=listed and not(np.isfinite(values[i]) and values[i]>0):
                missing.append({'date':str(day.date()),'symbol':symbol,'overlaps_known_split_window':overlap(day)})
            if str(day.date())==END: continue
            x=values[max(0,i-252):i+1];bad=~np.isfinite(x)|(x<=0)
            if day<listed: reason='not_yet_listed'
            elif i-origin+1<253: reason='initial_history_window_short'
            elif bad.any():
                dates=index[max(0,i-252):i+1][bad]
                reason='missing_price_window_with_action_overlap' if any(overlap(d) for d in dates) else 'missing_price_window_without_action_overlap'
            else:
                m=[math.log(x[-1]/x[-n-1])*252/n for n in (63,126,252)]
                reason='valid_positive_trend' if sum(m)>0 and sum(v>0 for v in m)>=2 else 'complete_history_trend_failed'
            row=lookup.loc[(str(day.date()),symbol)];flag=frames.eligible.at[day,symbol]
            require(pd.notna(flag),'资格数据缺失，不能将NaN当作真')
            usable=bool(flag) and reason=='valid_positive_trend'
            require(row.reason==reason and bool(row.eligible)==bool(flag) and bool(row.usable_signal)==usable,'独立历史分类不同')
            counts[reason]+=1;eligible_count+=usable
    expected_missing=pd.DataFrame(missing).sort_values(['symbol','date']).reset_index(drop=True)
    actual_missing=pd.read_csv(OUT/'missing_quotes.csv').sort_values(['symbol','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_missing,actual_missing)
    require(dict(counts)==result['history']['reason_counts'] and eligible_count==result['history']['eligible_usable_signal_sessions'], '候选分类汇总不同')
    inputs={**manifest['input_hashes'],**manifest['artifact_hashes']}
    inputs[(OUT/'completion.json').relative_to(BASE).as_posix()]=digest(OUT/'completion.json')
    inputs[Path(__file__).relative_to(BASE).as_posix()]=digest(Path(__file__))
    output={'passed':True,'input_hashes':inputs,'ledger_rows':days,'history_symbol_sessions':sum(counts.values()),
            'maximum_numeric_difference':maximum,'all_pnl_and_log_growth_components_reconciled':True,
            'cash_bridges_reconciled':True,'independent_history_categories_reconciled':True,'investment_admission':False}
    write_json_atomic(OUT/'verification.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='input_hashes'}))


if __name__=='__main__': audit()
