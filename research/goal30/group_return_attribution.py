"""冻结分组研究的逐日损益和现金等待归因；诊断，不产生新交易策略。"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.broad_action_evidence import READINESS
from research.goal30.group_diversification import feature_matrix, limit_weights
from research.goal30.primary_research_series import START, END, corrected_market, read
from research.goal30.run_dividend_payment_replay import GROUP
from research.goal30.turnover_budget import require
from research.goal30.validate_primary_execution import PrimaryMarketDays, REFERENCE

OUT=BASE/'output/goal30/group_attribution/20260914'
COMPONENTS=('cash_opportunity','invested_excess','trading_time','commission','slippage')


def pnl_components(previous_quantities,previous_close,close,actions,trades):
    """单位变换先归并，再按未复权价格计算损益；支付应收不产生第二份利润。"""
    splits={a['symbol']:a for a in actions if a['type']=='split'}
    market=rounding=0.
    for symbol,q in previous_quantities.items():
        ratio=splits[symbol]['ratio'] if symbol in splits else 1.
        market+=q*(ratio*close[symbol]-previous_close[symbol])
        if symbol in splits:
            require(q==splits[symbol]['quantity_before'],'拆分前数量不同')
            rounding+=(splits[symbol]['quantity_after']-q*ratio)*close[symbol]
    dividend=sum(a['amount'] for a in actions if a['type']=='accrue_dividend')
    timing=sum((1 if t['side']=='buy' else -1)*t['quantity']*(close[t['symbol']]-t['raw_price']) for t in trades)
    commission=sum(t['commission'] for t in trades)
    slippage=sum(t['slippage_cost'] for t in trades)
    return {'market_price_pnl':market,'split_rounding_pnl':rounding,'dividend_accrual_pnl':dividend,
            'trading_time_pnl':timing,'commission_cost':commission,'slippage_cost':slippage}


def log_link(rs,rb):
    require(math.isfinite(rs) and math.isfinite(rb) and rs>-1 and rb>-1,'对数归因收益无效')
    delta=rs-rb
    return 1/(1+rb) if abs(delta)<1e-14 else math.log1p(delta/(1+rb))/delta


def profit_attribution(data,frames,benchmark,initial):
    lookup={kind:defaultdict(list) for kind in ('trades','holdings','actions')}
    for kind in lookup:
        for item in data[kind]: lookup[kind][item['date']].append(item)
    previous_q={};previous_nav=previous_cash=initial;previous_claims=previous_value=0.
    rows=[];maximum_error=0.
    for i,report in enumerate(data['report']):
        day=report['date'];context=frames.context(day)
        pnl=pnl_components(previous_q,context['previous_close'],context['close'],lookup['actions'][day],lookup['trades'][day])
        absolute=sum(pnl[k] for k in ('market_price_pnl','split_rounding_pnl','dividend_accrual_pnl','trading_time_pnl'))-pnl['commission_cost']-pnl['slippage_cost']
        delta=report['account']-previous_nav;error=abs(absolute-delta);maximum_error=max(maximum_error,error)
        require(error<1e-7,'逐日金额归因不能解释经济净值变化：'+day)
        rs=delta/previous_nav;rb=float(benchmark.iloc[i]/benchmark.iloc[i-1]-1) if i else 0.
        components={'cash_opportunity':-(previous_cash+previous_claims)/previous_nav*rb,
            'invested_excess':(pnl['market_price_pnl']+pnl['split_rounding_pnl']+pnl['dividend_accrual_pnl']-previous_value*rb)/previous_nav,
            'trading_time':pnl['trading_time_pnl']/previous_nav,
            'commission':-pnl['commission_cost']/previous_nav,'slippage':-pnl['slippage_cost']/previous_nav}
        require(abs(sum(components.values())-(rs-rb))<1e-12,'相对收益归因不能相加')
        factor=log_link(rs,rb)
        rows.append({'date':day,'strategy_return':rs,'benchmark_return':rb,'previous_nav':previous_nav,
            'previous_cash_fraction':previous_cash/previous_nav,'previous_claim_fraction':previous_claims/previous_nav,
            'previous_invested_fraction':previous_value/previous_nav,'nav_change':delta,'pnl_error':error,**pnl,**components,
            **{k+'_linked_log':v*factor for k,v in components.items()}})
        previous_nav=report['account'];previous_cash=report['cash'];previous_claims=report['dividend_receivable'];previous_value=report['value']
        previous_q={h['symbol']:h['quantity'] for h in lookup['holdings'][day]}
    frame=pd.DataFrame(rows)
    growth=previous_nav/initial;benchmark_growth=benchmark.iloc[-1]/benchmark.iloc[0]
    years=(benchmark.index[-1]-benchmark.index[0]).days/365.25
    linked={k:float(frame[k+'_linked_log'].sum()) for k in COMPONENTS}
    relative_log=math.log(growth/benchmark_growth)
    require(abs(sum(linked.values())-relative_log)<1e-11,'跨期对数贡献不能解释相对增长')
    cs=growth**(1/years)-1;cb=benchmark_growth**(1/years)-1
    log_gap=math.log1p(cs)-math.log1p(cb)
    factor=(cs-cb)/log_gap if abs(log_gap)>1e-14 else 1+cb
    cagr_allocation={k:v/years*factor for k,v in linked.items()}
    require(abs(sum(cagr_allocation.values())-(cs-cb))<1e-11,'CAGR差额分配不闭合')
    annual=[]
    for year,group in frame.groupby(frame.date.str[:4]):
        value={k:float(group[k+'_linked_log'].sum()) for k in COMPONENTS}
        observed=float(np.log1p(group.strategy_return).sum()-np.log1p(group.benchmark_return).sum())
        require(abs(sum(value.values())-observed)<1e-11,'年度贡献不能解释相对增长')
        annual.append({'year':year,'relative_log_growth':observed,**value})
    return frame,{'strategy_cagr':cs,'benchmark_cagr':cb,'cagr_gap':cs-cb,'years':years,
        'maximum_daily_pnl_error_cny':maximum_error,'linked_relative_log_contributions':linked,
        'annualized_relative_log_contributions':{k:v/years for k,v in linked.items()},
        'cagr_gap_allocation_by_log_link':cagr_allocation,'annual_linked_log_contributions':annual,
        'total_commissions_cny':float(frame.commission_cost.sum()),'total_slippage_cny':float(frame.slippage_cost.sum()),
        'mean_previous_cash_fraction':float(frame.previous_cash_fraction.mean()),
        'mean_previous_claim_fraction':float(frame.previous_claim_fraction.mean()),
        'interpretation':'Descriptive allocation of observed relative returns. Cash opportunity uses benchmark returns; it does not assert SPY is tradable in the fixed pool, nor estimate a cost-free fully invested counterfactual. CAGR gap allocation uses a common logarithmic-link scale, not a unique causal attribution.'}


def history_diagnosis(frames,prices,features,events):
    listing={s:pd.Timestamp(r['listing_date']) for s,r in read(READINESS)['symbols'].items()}
    valid=np.isfinite(prices)&prices.gt(0);date_index=prices.index
    overlap=pd.DataFrame(False,index=date_index,columns=prices.columns)
    for event in events:
        if event['type']=='split':
            mask=(date_index>=event['record_date'])&(date_index<event['effective_date'])
            overlap.loc[mask,event['symbol']]=True
    missing_action=(~valid)&overlap
    has_missing=(~valid).rolling(253,min_periods=1).sum().gt(0)
    action_missing_window=missing_action.rolling(253,min_periods=1).sum().gt(0)
    first={s:date_index.get_loc(prices[s].first_valid_index()) for s in prices}
    rows=[]
    for day in date_index[(date_index>=START)&(date_index<=END)][:-1]:
        i=date_index.get_loc(day)
        for symbol in prices:
            origin=max(first[symbol],int(date_index.searchsorted(listing[symbol])))
            if day<listing[symbol]: reason='not_yet_listed'
            elif i-origin+1<253: reason='initial_history_window_short'
            elif has_missing.at[day,symbol]:
                reason='missing_price_window_with_action_overlap' if action_missing_window.at[day,symbol] else 'missing_price_window_without_action_overlap'
            elif pd.isna(features.at[day,symbol]): reason='complete_history_trend_failed'
            else: reason='valid_positive_trend'
            eligible=bool(frames.eligible.at[day,symbol])
            rows.append({'date':str(day.date()),'symbol':symbol,'reason':reason,'eligible':eligible,
                         'usable_signal':eligible and reason=='valid_positive_trend'})
    frame=pd.DataFrame(rows);counts=Counter(frame.reason)
    require(len(frame)==2427*30 and sum(counts.values())==len(frame),'候选诊断覆盖不完整')
    actual_missing=[]
    for symbol in prices:
        mask=(date_index>=max(pd.Timestamp(START),listing[symbol])) & (date_index<=END) & (~valid[symbol]).to_numpy()
        for day in date_index[mask]:
            actual_missing.append({'date':str(day.date()),'symbol':symbol,'overlaps_known_split_window':bool(overlap.at[day,symbol])})
    summary={'symbol_sessions':len(frame),'reason_counts':dict(counts),'eligible_usable_signal_sessions':int(frame.usable_signal.sum()),
        'observed_missing_symbol_days_in_backtest':len(actual_missing),
        'missing_days_overlapping_known_split_window':sum(r['overlaps_known_split_window'] for r in actual_missing),
        'mean_usable_symbols_per_day':float(frame.groupby('date').usable_signal.sum().mean()),
        'note':'Exclusive history categories across30 ETFs per signal day. A missing quote overlapping a split record-to-effective window is temporal association, not proof of suspension or causation. Rolling-window affected counts are symbol-days and can reflect the same missing quote repeatedly.'}
    return frame,pd.DataFrame(actual_missing),summary


def cash_bridge(data,prices,policy,variant):
    reports={r['date']:r for r in data['report']};rows=[];month_total=0.;original_selected=set()
    for signal in data['signals']:
        day=signal['date'];raw=signal['selection_state']['raw_weights'];total=sum(raw.values())
        if signal['monthly_review']:
            month_total=total;original_selected=set(raw)
        clipped=limit_weights(raw,prices.loc[:day],policy);first=sum(clipped.values())
        current=signal['current_weights']
        for s in clipped:
            if current.get(s,0)>0 and abs(clipped[s]-current[s])<policy['rebalance_band']: clipped[s]=current[s]
        capped_again=limit_weights(clipped,prices.loc[:day],policy);second=sum(capped_again.values())
        target=sum(signal['target_weights'].values());report=reports[day]
        invested=report['value']/(report['cash']+report['value'])
        pieces={'permanent_cash_reserve':1-policy['max_gross_weight'],
            'unfilled_slots_at_monthly_review':policy['max_gross_weight']-month_total,
            'removed_targets_since_monthly_review':month_total-total,
            'risk_and_position_caps':total-first,'drift_band_effect':first-second,
            'daily_turnover_budget':second-target,'target_to_current_holdings_gap':target-invested}
        require(abs(sum(pieces.values())-(1-invested))<1e-10,'现金比例桥不闭合')
        rows.append({'date':day,'liquid_cash_fraction':1-invested,**pieces,
            'monthly_review':signal['monthly_review'],'original_monthly_slots':len(original_selected),
            'remaining_monthly_slots':len(raw)})
    frame=pd.DataFrame(rows)
    return frame,{'mean_liquid_cash_fraction':float(frame.liquid_cash_fraction.mean()),
        'mean_components':{k:float(frame[k].mean()) for k in pieces},
        'days_with_monthly_targets_removed':int((frame.removed_targets_since_monthly_review>1e-10).sum()),
        'days_with_unfilled_initial_slots':int((frame.unfilled_slots_at_monthly_review>1e-10).sum()),
        'note':'Algebraic bridge from95% gross limit to contemporaneous actual exposure in liquid assets (cash+ETF market value; receivables excluded). It diagnoses target gaps, not causal profit loss. Targets are for next-session execution; signed residual can be negative while sells are pending.'}


def run():
    require(not (OUT/'completion.json').exists(),'不得覆盖已完成归因')
    OUT.mkdir(parents=True,exist_ok=True)
    proof=read(GROUP/'verification.json');registration=read(GROUP/'registration.json');results=read(GROUP/'results.json')
    inputs={**registration['input_hashes'],**proof['artifact_hashes']}
    for p in [GROUP/'verification.json',GROUP/'registration.json',GROUP/'results.json',READINESS,Path(__file__),
              Path(__file__).with_name('test_group_return_attribution.py')]: inputs[p.relative_to(BASE).as_posix()]=digest(p)
    for rel,h in inputs.items(): require(digest(BASE/rel)==h,'归因输入改变：'+rel)
    write_json_atomic(OUT/'registration.json',{'input_hashes':inputs,'created_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'source':GROUP.relative_to(BASE).as_posix(),'scope':'Five existing base cases only, descriptive attribution; no strategy choice or parameter tuning.'})
    frames=PrimaryMarketDays(corrected_market());prices=frames.market.wide('adj_close');features=feature_matrix(prices)
    benchmark=pd.read_csv(REFERENCE/'benchmark_fixed_control__base.csv',index_col=0,parse_dates=True).spy_total_cny
    history,missing,history_summary=history_diagnosis(frames,prices,features,registration['actions'])
    history.to_csv(OUT/'history_diagnosis.csv',index=False);missing.to_csv(OUT/'missing_quotes.csv',index=False)
    outcome={'cases':{},'history':history_summary,'investment_admission':False}
    for variant in registration['protocol']['variants']:
        key=variant['name']+'__base';item=results['cases'][key];path=GROUP/item['artifact']
        require(digest(path)==item['sha256'],'旧基础策略归档改变');data=read(path)
        dates=pd.DatetimeIndex(pd.to_datetime([r['date'] for r in data['report']]))
        require(dates.equals(benchmark.index),'基准与账本日期不同')
        attribution,summary=profit_attribution(data,frames,benchmark,registration['policy']['initial_capital'])
        require(abs(summary['strategy_cagr']-item['metrics']['strategy']['cagr'])<1e-12,'归因年化与已验收回测不同')
        bridge,bridge_summary=cash_bridge(data,prices,registration['policy'],variant)
        # 月内已退出且重新满足趋势/资格的标的仍必须等下月；只计原月选中标的。
        recovered_waits=0;selected=set()
        for signal in data['signals']:
            if signal['monthly_review']: selected=set(signal['selection_state']['raw_weights'])
            usable=set(features.columns[features.loc[pd.Timestamp(signal['date'])].notna()]) & set(signal['eligible']) - set(signal['exiting'])
            recovered_waits+=len((selected-set(signal['selection_state']['raw_weights']))&usable)
        bridge_summary['recovered_original_target_symbol_days_waiting_next_month']=recovered_waits
        attribution.to_csv(OUT/(key+'_pnl.csv'),index=False);bridge.to_csv(OUT/(key+'_cash_bridge.csv'),index=False)
        outcome['cases'][key]={'profit':summary,'cash_bridge':bridge_summary}
        print(json.dumps({'case':key,'cagr_gap_allocation':summary['cagr_gap_allocation_by_log_link'],
                          'pnl_error':summary['maximum_daily_pnl_error_cny'],'cash_bridge':bridge_summary['mean_components']}),flush=True)
    write_json_atomic(OUT/'results.json',outcome)
    for rel,h in inputs.items(): require(digest(BASE/rel)==h,'归因中输入改变：'+rel)
    outputs={p.relative_to(BASE).as_posix():digest(p) for p in OUT.iterdir() if p.is_file()}
    write_json_atomic(OUT/'completion.json',{'input_hashes':inputs,'artifact_hashes':outputs,'cases':5,
        'ledger_rows':5*2428,'history_symbol_sessions':2427*30,'passed':True,'investment_admission':False})


if __name__=='__main__': run()
