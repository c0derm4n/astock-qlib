"""收尾验证：成本加压 + 换手率检查（对当前生产信号 = 无过滤 TopK）。

1. 成本加压：滑点 0/5/10/20bp 重跑同一信号，确认收益不是靠低成本假设撑起来的
   （backtest-expert 方法论：策略要在悲观假设下存活）；
2. 换手率检查：日均/年化单边换手、成本年拖累，评估每日调仓+N_DROP 的实际磨损。

不重训模型，直接复用 output/predictions_raw.pkl。用法：
    python -m src.stress
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

import config


def _bt_report(signal: pd.Series, slippage: float) -> pd.DataFrame:
    """与 train 同参回测（滑点可调），返回完整日度报告(return/cost/bench/turnover)。"""
    from src.utils import patch_qlib_deterministic
    patch_qlib_deterministic()  # 固定持仓列表顺序，保证回测跨进程可复现
    dates = signal.index.get_level_values("datetime").unique().sort_values()
    strategy = TopkDropoutStrategy(signal=signal, topk=config.TOPK, n_drop=config.N_DROP,
                                       hold_thresh=getattr(config, "HOLD_THRESH", 1))
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    pd_dict, _ = backtest(
        start_time=dates[0], end_time=dates[-2], strategy=strategy, executor=executor,
        account=100_000_000, benchmark=config.BENCHMARK_SYMBOL,
        exchange_kwargs={
            "freq": "day", "limit_threshold": config.LIMIT_THRESHOLD,
            "deal_price": getattr(config, "DEAL_PRICE", "close"),
            "open_cost": config.OPEN_COST + slippage,
            "close_cost": config.CLOSE_COST + slippage, "min_cost": config.MIN_COST,
        },
    )
    return list(pd_dict.values())[0][0]


def _metrics(r: pd.Series) -> dict:
    nav = (1.0 + r.fillna(0.0)).cumprod()
    ann = nav.iloc[-1] ** (252.0 / len(r)) - 1.0
    vol = r.std() * np.sqrt(252.0)
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"累计": nav.iloc[-1] - 1.0, "年化": ann,
            "Sharpe": ann / vol if vol > 0 else float("nan"), "最大回撤": mdd}


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    from src.overlay import _drop_qdii

    signal = _drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl"))

    # ---- 1) 成本加压：滑点 0/5/10/20bp ----
    rows = {}
    rep5 = None
    for bps in (0.0, 0.0005, 0.0010, 0.0020):
        rep = _bt_report(signal, bps)
        if bps == 0.0005:
            rep5 = rep
        rows[f"滑点{bps*1e4:.0f}bp"] = _metrics(rep["return"] - rep["cost"])
    tab = pd.DataFrame(rows).T
    for c in ("累计", "年化", "最大回撤"):
        tab[c] = tab[c].map(lambda x: f"{x:+.1%}")
    tab["Sharpe"] = tab["Sharpe"].map(lambda x: f"{x:.2f}")
    print("\n===== 成本加压（无过滤TopK，walk-forward样本外）=====")
    print(tab.to_string())

    # ---- 2) 换手率与成本拖累（基于 5bp 基线报告）----
    to = rep5["turnover"].fillna(0.0)      # 日单边换手率(占组合市值)
    cost = rep5["cost"].fillna(0.0)        # 日成本(占组合市值)
    years = len(to) / 252.0
    print("\n===== 换手率检查（滑点5bp基线）=====")
    print(f"日均单边换手：{to.mean():.2%}   年化单边换手：{to.mean()*252:.0f}%")
    print(f"平均持有期（1/日换手）：约 {1.0/to[to>0].mean():.0f} 个交易日")
    print(f"成本年拖累：{cost.mean()*252:.2%}/年   区间总成本：{cost.sum():.1%}"
          f"（{years:.1f}年）")
    yto = (to.groupby(to.index.year).mean() * 252).map(lambda x: f"{x:.0f}%")
    print("逐年年化单边换手：")
    print(yto.to_string())


if __name__ == "__main__":
    main()
