"""估值带信号：隐含滚动股息率、股债利差、滚动百分位与档位映射。"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIVIDEND_WINDOW = 252
PCT_WINDOW = 756
PCT_MIN = 252
MARKET_MA = 250
VOL_WINDOW = 60
CASH_CAP = 0.95


def indicators(panel):
    """只用当日及更早数据构造估值指标与风险状态。"""
    red, price = panel.red.astype(float), panel.price.astype(float)
    y10 = panel.yield10.astype(float)
    implied = red / red.shift(1) * price.shift(1) - price
    implied.iloc[0] = np.nan
    dividend = implied.rolling(DIVIDEND_WINDOW, min_periods=DIVIDEND_WINDOW).sum() / price * 100
    erp = dividend - y10
    dividend_pct = dividend.rolling(PCT_WINDOW, min_periods=PCT_MIN).rank(pct=True) * 100
    erp_pct = erp.rolling(PCT_WINDOW, min_periods=PCT_MIN).rank(pct=True) * 100
    ma = price.rolling(MARKET_MA, min_periods=MARKET_MA).mean()
    vol = price.pct_change(fill_method=None).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252)
    deep = (price < ma * 0.96) & (ma < ma.shift(20))
    risk = (deep | (vol > 0.20)).fillna(False)
    return pd.DataFrame({'dividend_yield': dividend, 'dividend_pct': dividend_pct,
                         'erp': erp, 'erp_pct': erp_pct, 'ma250': ma,
                         'vol60': vol, 'risk': risk}, index=panel.index)


def band_weight(value, table):
    """按登记档位表取目标权重；表按阈值降序，最后一档为兜底。"""
    assert table and all(table[i][0] > table[i + 1][0] for i in range(len(table) - 1))
    for threshold, weight in table:
        if float(value) >= float(threshold):
            return float(weight)
    return float(table[-1][1])


def target_weights(panel, signals, candidate, tables):
    """把档位权重转换为红利/国债目标权重矩阵。"""
    index = panel.index
    target = pd.DataFrame(0., index=index, columns=['red', 'bond'])
    kind = candidate['indicator']
    if kind == 'hold':
        target['red'] = 0.95
    elif kind == 'bond':
        target['bond'] = 0.95
    elif kind == 'static':
        target['red'], target['bond'] = float(candidate['red']), 0.95 - float(candidate['red'])
    else:
        column = signals[kind]
        table = tables[candidate['table']]
        weight = pd.Series([band_weight(v, table) if np.isfinite(v) else np.nan
                            for v in column.to_numpy()], index=index)
        weight = weight.ffill().fillna(0.)
        assert weight.loc[index >= index[0]].notna().all()
        if candidate.get('guard'):
            weight = np.where(signals.risk.to_numpy(), np.minimum(weight.to_numpy(), 0.30),
                              weight.to_numpy())
            weight = pd.Series(weight, index=index)
        weight = weight.clip(0., CASH_CAP)
        target['red'] = weight
        if candidate.get('idle', 'bond') == 'bond':
            target['bond'] = (CASH_CAP - weight).clip(lower=0.)
    # 国债上市前不可持有；该部分保持现金，不造上市前收益。
    target.loc[~panel.bond_listed.astype(bool), 'bond'] = 0.
    assert target.ge(-1e-12).all().all()
    assert target.sum(axis=1).le(CASH_CAP + 1e-12).all()
    return target
