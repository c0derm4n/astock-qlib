"""D1(持有10日+平滑3日) 的“打死它”验证：参数平原 + 逐年分解。

turnover.py 初筛中 D1 累计 +244.7%(5bp)、+221.4%(10bp)，是基线的12倍，
好得反常。按 backtest-expert 方法论做两项压力验证：

1. 参数平原：hold ∈ {5,8,10,12,15} × smooth ∈ {1,2,3,4,5} 共25组合 @5bp，
   若 (10,3) 周围组合普遍优秀 → 平原(可信)；若孤峰 → 过拟合(否决)。
2. 逐年分解：D1 的逐年收益/回撤，确认不是靠单一行情(如2024Q4)撑起，
   并暴露阴跌年(2022)硬扛风险。

用法：
    python -m src.turnover_grid
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN

import config
from src.stress import _metrics
from src.turnover import _bt, _smooth


def _yearly(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).groupby(r.index.year).prod() - 1.0


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    from src.overlay import _drop_qdii

    raw = _drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl"))

    holds = [5, 8, 10, 12, 15]
    smooths = [1, 2, 3, 4, 5]

    cum = pd.DataFrame(index=[f"h{h}" for h in holds],
                       columns=[f"s{s}" for s in smooths], dtype=float)
    shp = cum.copy()
    mdd = cum.copy()
    tnv = cum.copy()

    for h in holds:
        for s in smooths:
            sig = _smooth(raw, s) if s > 1 else raw
            rep = _bt(sig, 0.0005, topk=6, n_drop=1, hold_thresh=h)
            r = rep["return"] - rep["cost"]
            m = _metrics(r)
            cum.loc[f"h{h}", f"s{s}"] = m["累计"]
            shp.loc[f"h{h}", f"s{s}"] = m["Sharpe"]
            mdd.loc[f"h{h}", f"s{s}"] = m["最大回撤"]
            tnv.loc[f"h{h}", f"s{s}"] = rep["turnover"].fillna(0.0).mean()
        print(f"[done] hold={h}", flush=True)

    pd.set_option("display.width", 160)
    print("\n===== 参数平原 @5bp：累计收益 =====")
    print(cum.applymap(lambda x: f"{x:+.0%}").to_string())
    print("\n===== Sharpe =====")
    print(shp.applymap(lambda x: f"{x:.2f}").to_string())
    print("\n===== 最大回撤 =====")
    print(mdd.applymap(lambda x: f"{x:.0%}").to_string())
    print("\n===== 日均换手 =====")
    print(tnv.applymap(lambda x: f"{x:.0%}").to_string())

    # ---- 逐年分解：D1 及两个邻居 ----
    print("\n===== 逐年收益分解 @5bp（策略 | 沪深300）=====")
    bench = None
    for name, h, s in [("D1 h10/s3", 10, 3), ("h8/s3", 8, 3), ("h10/s4", 10, 4)]:
        sig = _smooth(raw, s) if s > 1 else raw
        rep = _bt(sig, 0.0005, topk=6, n_drop=1, hold_thresh=h)
        r = rep["return"] - rep["cost"]
        y = _yearly(r)
        if bench is None:
            bench = _yearly(rep["bench"].fillna(0.0))
        nav = (1.0 + r.fillna(0.0)).cumprod()
        mdd_by_y = nav.groupby(nav.index.year).apply(
            lambda v: (v / v.cummax() - 1.0).min())
        print(f"\n--- {name}（累计 {(1+r.fillna(0)).prod()-1:+.0%}）---")
        for yy in y.index:
            b = bench.get(yy, np.nan)
            print(f"  {yy}: 策略 {y[yy]:+.1%}   年内最大回撤 {mdd_by_y[yy]:.1%}"
                  f"   沪深300 {b:+.1%}")


if __name__ == "__main__":
    main()
