"""强制排除纳指；复用已验证的因果信号、风险与成交引擎。"""
from __future__ import annotations
import copy
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import universe
from research.sp500.explore import frozen_market, signal_matrix, run_candidate
from src.portfolio import risk_group

EXCLUDED = "SH513100"
SYMBOLS = frozenset(("SH" if code.startswith(("5", "6")) else "SZ") + code
                    for code in universe.UNIVERSE if code != "513100")


def market30(source):
    market = frozen_market(source, include_qdii=True)
    market.bars = market.bars[market.bars.symbol.isin(SYMBOLS)].copy()
    market.eligibility = market.eligibility[market.eligibility.symbol.isin(SYMBOLS)].copy()
    for name in ("features", "labels", "label_ends"):
        value = getattr(market, name)
        setattr(market, name, value[value.index.get_level_values("instrument").isin(SYMBOLS)].copy())
    for name in ("nav", "meta"):
        frame = getattr(market, name)
        setattr(market, name, frame[frame.symbol.isin(SYMBOLS)].copy())
    market.audit = {**market.audit, "symbols": len(SYMBOLS), "bars": len(market.bars),
                    "excluded_symbols": [EXCLUDED]}
    if len(SYMBOLS) != 30 or set(market.bars.symbol.unique()) != SYMBOLS:
        raise ValueError("本研究要求原池剔除513100后恰好30只ETF")
    return market


def validate_candidate(candidate):
    symbols = set(candidate.get("weights", {}))
    symbols.update(candidate.get("signal_symbols", []))
    for original, alternatives in candidate.get("fallbacks", {}).items():
        symbols.add(original)
        symbols.update(alternatives)
    if symbols - SYMBOLS:
        raise ValueError(f"候选包含池外/被排除标的：{sorted(symbols - SYMBOLS)}")
    if candidate["family"] not in ("fixed", "rotation", "hybrid"):
        raise ValueError("本研究仅使用独立重新回放的候选")
    if candidate.get("signal") == "lightgbm":
        raise ValueError("旧模型训练池含纳指，本轮不复用其预测")


def signals(market, candidate):
    result = unfiltered_signals(market, candidate)
    if candidate.get("trend_window"):
        price = market.wide("adj_close")
        result = result.where(price > price.rolling(candidate["trend_window"]).mean())
    if candidate.get("signal_scope") == "macro":
        allowed = {s for s in SYMBOLS if risk_group(s) in ("broad", "bond", "commodity", "qdii")}
        result.loc[:, ~result.columns.isin(allowed)] = np.nan
    if "signal_symbols" in candidate:
        result.loc[:, ~result.columns.isin(candidate["signal_symbols"])] = np.nan
    return result


def unfiltered_signals(market, candidate):
    if candidate.get('signal') == 'frozen_model':
        base = Path(__file__).resolve().parents[2]
        path = base/candidate['predictions']
        manifest = json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['artifact_hashes'][path.name]:
            raise ValueError('滚动模型预测文件已变化')
        if not manifest['all_label_ends_before_predictions']:
            raise ValueError('滚动模型存在未成熟标签')
        result = pd.read_parquet(path)
        if set(result.columns) != SYMBOLS:
            raise ValueError('模型训练/预测池必须严格30只')
        return result.reindex(index=market.calendar, columns=market.wide('adj_close').columns)
    if candidate.get("signal") == "pullback":
        price = market.wide("adj_close")
        n = candidate['lookbacks'][0]
        vol = price.pct_change(fill_method=None).rolling(60, min_periods=20).std().clip(lower=.002)
        return (-(price/price.shift(n)-1)/vol).replace([np.inf, -np.inf], np.nan)
    if candidate.get("signal") != "regression_trend":
        return signal_matrix(market, candidate, None)
    # 等权OLS拟合过去N日对数收盘；年化斜率乘R平方，压低噪声趋势。
    price = market.wide("adj_close")
    n = candidate["lookbacks"][0]
    if n < 3:
        raise ValueError("回归窗口至少3日")
    x = np.arange(n, dtype=float)
    x -= x.mean()
    xx = float(x @ x)
    values = np.log(price.to_numpy())
    windows = np.lib.stride_tricks.sliding_window_view(values, n, axis=0)
    centered = windows - windows.mean(axis=-1, keepdims=True)
    xy = (centered * x).sum(axis=-1)
    yy = (centered ** 2).sum(axis=-1)
    r2 = np.divide(xy ** 2, xx * yy, out=np.zeros_like(xy), where=yy > 1e-15)
    score = xy / xx * 252 * r2
    result = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    result.iloc[n-1:] = score
    return result.replace([np.inf, -np.inf], np.nan)


def execute(market, candidate, policy, start, **stress):
    validate_candidate(candidate)
    if set(market.bars.symbol.unique()) != SYMBOLS:
        raise ValueError("回放行情必须严格限制为30只")
    scores = signals(market, candidate)
    if EXCLUDED in scores.columns:
        raise ValueError("信号混入纳指")
    tested_market = market
    if candidate.get('daily_trend_exit'):
        # 策略趋势门槛逐日作用，不等到下一排名调仓日；国债作为停泊资产豁免。
        tested_market = copy.copy(market)
        price = market.wide('adj_close')
        trend = price.gt(price.rolling(candidate['daily_trend_exit']).mean())
        trend['SH511260'] = True
        gate = trend.stack(future_stack=True)
        index = pd.MultiIndex.from_frame(market.eligibility[['date','symbol']])
        tested_market.eligibility = market.eligibility.copy()
        tested_market.eligibility['eligible'] &= gate.reindex(index).fillna(False).to_numpy()
    report, trades, issues = run_candidate(tested_market, scores, candidate, policy, start, **stress)
    if not trades.empty and set(trades.symbol) - SYMBOLS:
        raise ValueError("成交存在池外标的")
    return report, trades, issues
