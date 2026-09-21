"""在原目标预算内重新分配仓位截断资金；保留原选择与退出时点。"""
from __future__ import annotations

import math

import numpy as np

from research.ex_nasdaq.core import SYMBOLS
from research.goal30 import group_diversification as legacy
from research.goal30.turnover_budget import require
from src.portfolio import risk_group


def capped_proportional(weights,caps,budget):
    """活跃集比例分配；每轮固定触及容量的项，再按剩余原比例分配。"""
    require(set(weights)==set(caps),'比例与容量标的不同')
    require(math.isfinite(budget) and budget>=0,'分配预算无效')
    require(all(math.isfinite(w) and w>0 for w in weights.values()) and
            all(math.isfinite(c) and c>=0 for c in caps.values()),'分配比例或容量无效')
    if not weights: return {}
    remaining=min(float(budget),math.fsum(caps.values()))
    pending=sorted(weights);result={}
    for _ in range(len(weights)+1):
        if not pending: break
        total=math.fsum(weights[s] for s in pending)
        provisional={s:remaining*weights[s]/total for s in pending}
        saturated=[s for s in pending if provisional[s]>caps[s]]
        if not saturated:
            result.update(provisional);pending=[];break
        for s in saturated:
            result[s]=float(caps[s]);remaining-=caps[s]
        remaining=max(remaining,0.)
        pending=[s for s in pending if s not in saturated]
    require(not pending,'比例分配未收敛')
    require(all(-1e-12<=result[s]<=caps[s]+1e-12 for s in result),'比例分配越界')
    require(abs(math.fsum(result.values())-min(budget,math.fsum(caps.values())))<1e-10,'容量分配不守恒')
    return {s:result[s] for s in sorted(result)}


def position_redistribution(weights,policy):
    require(set(weights).issubset(SYMBOLS),'目标出现池外标的')
    require(all(math.isfinite(w) and w>=0 for w in weights.values()),'原始权重无效')
    raw={s:float(w) for s,w in weights.items() if w>1e-10}
    if not raw: return {}
    require(all(math.isfinite(policy[k]) and 0<policy[k]<=1 for k in
                ('max_weight','max_group_weight','max_gross_weight')),'仓位限制无效')
    groups={g:sorted(s for s in raw if risk_group(s)==g) for g in sorted({risk_group(s) for s in raw})}
    group_weights={g:math.fsum(raw[s] for s in symbols) for g,symbols in groups.items()}
    capacities={g:min(policy['max_group_weight'],len(symbols)*policy['max_weight']) for g,symbols in groups.items()}
    budget=min(math.fsum(raw.values()),policy['max_gross_weight'])
    group_allocation=capped_proportional(group_weights,capacities,budget)
    result={}
    for group,symbols in groups.items():
        result.update(capped_proportional({s:raw[s] for s in symbols},{s:policy['max_weight'] for s in symbols},group_allocation[group]))
    return {s:result[s] for s in sorted(result) if result[s]>1e-10}


def decision(previous,scores,prices,eligible,exiting,current,policy,variant):
    state,_,trace=legacy.decision(previous,scores,prices,eligible,exiting,current,policy,variant)
    raw=state['raw_weights']
    baseline=legacy.limit_weights(raw,prices,policy)
    filled=position_redistribution(raw,policy)
    desired=legacy.limit_weights(filled,prices,policy)
    first=dict(desired)
    for s in desired:
        if current.get(s,0)>0 and abs(desired[s]-current[s])<policy['rebalance_band']:
            desired[s]=current[s]
    desired=legacy.limit_weights(desired,prices,policy)
    before_turnover=dict(desired)
    sells=sum(max(0.,w-desired.get(s,0)) for s,w in current.items())
    buys={s:max(0.,w-current.get(s,0)) for s,w in desired.items()}
    scale=min(1.,max(0.,policy['max_daily_turnover']-sells)/max(sum(buys.values()),1e-12))
    targets={s:min(w,current.get(s,0))+buys[s]*scale for s,w in desired.items()}
    targets={s:w for s,w in targets.items() if w>1e-10}
    return state,targets,{**trace,'position_redistributed_weights':filled,'first_risk_limited_weights':first,
        'before_turnover_weights':before_turnover,'same_input_legacy_weights':baseline,
        'gross_redistribution_delta_after_volatility_limit':sum(first.values())-sum(baseline.values())}
