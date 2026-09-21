"""独立二分算法核查有上限分配；按实际账户和原价前缀核查每日目标。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.goal30 import group_diversification as legacy
from research.goal30.turnover_budget import require
from src.portfolio import risk_group


def bisect_allocation(weights,caps,budget):
    if not weights: return {}
    target=min(budget,sum(caps.values()));low=0.;high=max(caps[s]/weights[s] for s in weights)
    for _ in range(90):
        mid=(low+high)/2
        filled=sum(min(caps[s],mid*weights[s]) for s in weights)
        if filled<target: low=mid
        else: high=mid
    return {s:min(caps[s],(low+high)/2*weights[s]) for s in weights}


def independent_positions(raw,policy):
    raw={s:w for s,w in raw.items() if w>1e-10}
    if not raw: return {}
    grouped={}
    for s in raw: grouped.setdefault(risk_group(s),[]).append(s)
    group_weights={g:sum(raw[s] for s in members) for g,members in grouped.items()}
    caps={g:min(len(members)*policy['max_weight'],policy['max_group_weight']) for g,members in grouped.items()}
    target=min(sum(raw.values()),policy['max_gross_weight'])
    budgets=bisect_allocation(group_weights,caps,target);result={}
    for g,members in grouped.items():
        result.update(bisect_allocation({s:raw[s] for s in members},{s:policy['max_weight'] for s in members},budgets[g]))
    return {s:w for s,w in result.items() if w>1e-10}


def independent_risk_limit(weights,prices,policy):
    result={s:min(w,policy['max_weight']) for s,w in weights.items() if w>1e-10}
    for group in sorted({risk_group(s) for s in result}):
        members=[s for s in result if risk_group(s)==group]
        shrink=min(1.,policy['max_group_weight']/sum(result[s] for s in members))
        for s in members: result[s]*=shrink
    if not result: return {},0.
    shrink=min(1.,policy['max_gross_weight']/sum(result.values()))
    result={s:w*shrink for s,w in result.items()}
    symbols=sorted(result);x=prices[symbols].tail(64).to_numpy(float)
    require(x.shape==(64,len(symbols)) and np.isfinite(x).all() and (x>0).all(),'独立风险窗口缺价')
    changes=x[1:]/x[:-1]-1;centered=changes-changes.mean(axis=0)
    covariance=(centered.T@centered)/(len(changes)-1)*252
    matrix=.75*covariance+.25*np.diag(np.maximum(np.diag(covariance),1e-8))
    vector=np.array([result[s] for s in symbols]);vol=math.sqrt(float(vector@matrix@vector))
    factor=min(1.,.15/max(vol,1e-12));result={s:w*factor for s,w in result.items()}
    return result,vol*factor


def independent_targets(raw,prices,current,policy):
    positions=independent_positions(raw,policy);first,vol1=independent_risk_limit(positions,prices,policy)
    desired=dict(first)
    for s in desired:
        if current.get(s,0)>0 and abs(desired[s]-current[s])<policy['rebalance_band']: desired[s]=current[s]
    desired,vol2=independent_risk_limit(desired,prices,policy)
    before_turnover=dict(desired)
    sold=sum(max(0.,current.get(s,0)-desired.get(s,0)) for s in current)
    increments={s:max(0.,desired[s]-current.get(s,0)) for s in desired}
    fraction=min(1.,max(0.,policy['max_daily_turnover']-sold)/max(sum(increments.values()),1e-12))
    targets={s:min(desired[s],current.get(s,0))+fraction*increments[s] for s in desired}
    targets={s:w for s,w in targets.items() if w>1e-10}
    return targets,positions,first,before_turnover,max(vol1,vol2)


def same(actual,expected,label):
    require(set(actual)==set(expected) and all(abs(actual[s]-expected[s])<1e-9 for s in expected),label)


def forecast_volatility(weights,prices):
    if not weights: return 0.
    symbols=sorted(weights);x=prices[symbols].tail(64).to_numpy(float)
    require(x.shape==(64,len(symbols)) and np.isfinite(x).all() and (x>0).all(),'预测风险窗口缺失')
    returns=x[1:]/x[:-1]-1;centered=returns-returns.mean(axis=0)
    covariance=centered.T@centered/62*252
    covariance=.75*covariance+.25*np.diag(np.maximum(np.diag(covariance),1e-8))
    w=np.array([weights[s] for s in symbols])
    return math.sqrt(float(w@covariance@w))


def verify_signals(frames,prices,bank,registered,variant,data):
    reports={r['date']:r for r in data['report']};holdings={};trades={}
    for row in data['holdings']: holdings.setdefault(row['date'],[]).append(row)
    for row in data['trades']: trades.setdefault(row['date'],[]).append(row)
    require([r['date'] for r in data['signals']]==list(reports)[:-1]==list(bank),'信号日期不完整')
    state={};exiting=set();reviews=0;volatility=0.;extra_days=0;extra_sum=0.;filled_cases=0
    post_budget_volatility=0.;post_budget_exceedances=0
    for row in data['signals']:
        day=row['date'];hold=holdings.get(day,[]);qty={v['symbol']:v['quantity'] for v in hold}
        orders={(o['symbol'],o['side']):o for o in data['pending'][day]}
        for trade in trades.get(day,[]):
            if trade['side']=='sell' and orders[(trade['symbol'],'sell')]['target_weight']<=1e-9: exiting.add(trade['symbol'])
        exiting.intersection_update(qty);require(sorted(exiting)==row['exiting'],'退出状态不能从成交重建')
        report=reports[day];current={v['symbol']:v['value']/(report['cash']+report['value']) for v in hold}
        same(row['current_weights'],current,'持仓权重不同')
        allowed=sorted(frames.eligible.columns[frames.eligible.loc[pd.Timestamp(day)].fillna(False)].tolist())
        require(allowed==row['eligible'],'信号资格不同')
        prefix=prices.loc[:day]
        state,_,trace=legacy.decision(state,bank[day],prefix,allowed,exiting,current,registered['policy'],variant)
        require(state['month']==row['selection_state']['month'] and trace['monthly_review']==row['monthly_review'],'月度选择时间不同')
        same(state['raw_weights'],row['selection_state']['raw_weights'],'原始月度选择不同')
        same(trace['unconstrained_risk_weights'],row['unconstrained_risk_weights'],'原始风险比例不同')
        target,positions,first,pre_turnover,vol=independent_targets(state['raw_weights'],prefix,current,registered['policy'])
        for actual,expected,label in [(row['target_weights'],target,'目标'),(row['position_redistributed_weights'],positions,'资金重分配'),
                                     (row['first_risk_limited_weights'],first,'第一次波动约束'),(row['before_turnover_weights'],pre_turnover,'换手前目标')]:
            same(actual,expected,'独立二分/直接协方差不同：'+label)
        old=legacy.limit_weights(state['raw_weights'],prefix,registered['policy'])
        same(row['same_input_legacy_weights'],old,'同输入原方案不同')
        delta=sum(first.values())-sum(old.values())
        require(abs(delta-row['gross_redistribution_delta_after_volatility_limit'])<1e-9,'资金使用变化不同')
        require(set(target).issubset(allowed) and not(set(target)&exiting),'不合资格资产仍获权重')
        require(vol<=.15+1e-10 and all(w<=.35+1e-10 for w in target.values()) and sum(target.values())<=.95+1e-10,'目标限制违反')
        for group in {risk_group(s) for s in target}:
            require(sum(w for s,w in target.items() if risk_group(s)==group)<=.70+1e-10,'组别目标超限')
        volatility=max(volatility,vol);reviews+=trace['monthly_review'];extra_days+=delta>1e-10;extra_sum+=delta;filled_cases+=bool(positions)
        target_vol=forecast_volatility(target,prefix)
        post_budget_volatility=max(post_budget_volatility,target_vol);post_budget_exceedances+=target_vol>.15+1e-10
    for trade in data['trades']:
        if trade['side']=='buy': require(bool(frames.eligible.at[pd.Timestamp(trade['signal_date']),trade['symbol']]),'成交缺少信号日资格')
        require(trade['participation']<=registered['policy']['max_adv_participation']+1e-12,'成交超过参与率')
    return {'signals_verified':len(data['signals']),'monthly_reviews':reviews,'nonempty_allocation_days':filled_cases,
        'maximum_checked_pre_turnover_forecast_volatility':volatility,'days_with_increased_first_target_gross':int(extra_days),
        'mean_first_target_gross_change_same_input':extra_sum/len(data['signals']),
        'maximum_post_budget_target_forecast_volatility':post_budget_volatility,
        'post_budget_target_volatility_above_15pct_days':int(post_budget_exceedances),
        'independent_bisection_and_covariance_verified':True}
