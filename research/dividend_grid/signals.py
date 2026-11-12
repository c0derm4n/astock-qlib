"""因果指标与周度目标状态；风险优先，恢复与网格分开。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GridState:
    stage: int = 0
    risk: bool = False
    recovery_count: int = 0


def candidates():
    base=dict(core=.60,grid=False,risk=False,bond=False,ma=200,
              buys=[-.05,-.10],sells=[.06,.08],vol_entry=.20,vol_exit=.18,
              recover=2,relative=False,volgrid=False,hold95=False)
    rows=[]
    definitions=[('hold95_cash',dict(hold95=True)),('static60_cash',{}),
        ('static60_bond',dict(bond=True)),('grid_cash',dict(grid=True)),
        ('grid_bond',dict(grid=True,bond=True)),('risk60_cash',dict(risk=True)),
        ('risk60_bond',dict(risk=True,bond=True)),('grid_risk_cash',dict(grid=True,risk=True)),
        ('grid_risk_bond',dict(grid=True,risk=True,bond=True))]
    primary={**base,**dict(grid=True,risk=True,bond=True)}
    for name,overrides in definitions:
        rows.append({**base,**overrides,'id':name})
    for suffix,overrides in [
        ('core50',dict(core=.50)),('ma180',dict(ma=180)),('ma250',dict(ma=250)),
        ('narrow',dict(buys=[-.04,-.08],sells=[.05,.07])),
        ('wide',dict(buys=[-.06,-.12],sells=[.07,.09])),
        ('vol18',dict(vol_entry=.18,vol_exit=.16)),
        ('vol22',dict(vol_entry=.22,vol_exit=.20)),
        ('recover1',dict(recover=1)),('recover3',dict(recover=3)),
        ('volgrid',dict(volgrid=True))]:
        rows.append({**primary,**overrides,'id':'grid_risk_bond_'+suffix})
    rows.append({**primary,'relative':True,'id':'relative_risk_bond'})
    return rows


def indicators(proxy,ma):
    p=proxy.red
    out=pd.DataFrame({'price':p,'ma_grid':p.rolling(ma,min_periods=ma).mean(),
                      'ma_risk':p.rolling(250,min_periods=250).mean()},index=proxy.index)
    out['slope']=out.ma_risk/out.ma_risk.shift(20)-1
    out['bias']=p/out.ma_grid-1
    out['vol']=p.pct_change(fill_method=None).rolling(60,min_periods=60).std(ddof=1)*np.sqrt(252)
    out['relative40']=p/p.shift(40)-proxy.csi/proxy.csi.shift(40)
    return out


def decide(state,row,candidate):
    required=['price','ma_grid','ma_risk','slope','bias','vol']
    if not np.isfinite([row[k] for k in required]).all():
        raise ValueError('指标预热不完整，不允许补未来数据')
    core=candidate['core']
    levels=[core,(core+.90)/2,.95]
    if candidate['hold95']:
        return .95,'hold95'
    bad=((row['price']<row['ma_risk']*.96 and row['slope']<0)
         or row['vol']>candidate['vol_entry'])
    if candidate['risk'] and bad:
        state.stage=0;state.risk=True;state.recovery_count=0
        return core/2,'risk_cap'
    if candidate['risk'] and state.risk:
        good=(row['price']>row['ma_risk']*1.01 and row['slope']>=0
              and row['vol']<candidate['vol_exit'])
        state.recovery_count=state.recovery_count+1 if good else 0
        if state.recovery_count>=candidate['recover']:
            state.risk=False;state.recovery_count=0;state.stage=0
            return core,'risk_recovery'
        return core/2,'risk_wait'
    if not candidate['grid']:
        return core,'static_core'
    stage=state.stage
    scale=float(np.clip(row['vol']/.15,.75,1.5)) if candidate['volgrid'] else 1.
    if candidate['relative']:
        if not np.isfinite(row['relative40']):
            raise ValueError('相对收益缺少全收益对照')
        buy=stage<2 and row['bias']<0 and row['relative40']<=[-.01,-.03][min(stage,1)]
        sell=stage>0 and row['relative40']>=[.06,.05][stage-1]
    else:
        buy=stage<2 and row['bias']<=candidate['buys'][min(stage,1)]*scale
        sell=stage>0 and row['bias']>=candidate['sells'][2-stage]*scale
    if sell:
        state.stage-=1;reason='grid_reduce'
    elif buy:
        state.stage+=1;reason='grid_add'
    else:
        reason='grid_hold'
    return levels[state.stage],reason


def targets(proxy,candidate,start,week_end=False):
    features=indicators(proxy,candidate['ma'])
    path=proxy.loc[start:]
    periods=path.index.to_period('W-SUN')
    schedule=(np.r_[periods[:-1]!=periods[1:],True] if week_end else
              np.r_[True,periods[1:]!=periods[:-1]])
    state=GridState();records=[]
    for day in path.index[schedule]:
        w,reason=decide(state,features.loc[day],candidate)
        bond=(max(0.,.95-w) if candidate['bond'] and bool(path.loc[day,'bond_listed']) else 0.)
        records.append({'date':day,'red':w,'bond':bond,'stage':state.stage,
            'risk':state.risk,'recovery_count':state.recovery_count,'reason':reason,
            **features.loc[day].to_dict()})
    return pd.DataFrame(records).set_index('date')
