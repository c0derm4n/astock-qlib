"""实验C：降换手 —— 模型打分不动，只改调仓规则（复用信号，无需重训）。

动机（src.stress 收尾验证结论）：基线日均换手约31%(qlib双边口径)，
成本年拖累 6.34%/年；零成本年化 +7.8% → 5bp 下 +3.6% → 10bp 下 -0.4%，
毛利近半被交易磨损吃掉，10bp 悲观假设下策略归零。

对照变体（全部复用 output/predictions_raw.pkl 的 walk-forward 样本外信号）：
  V0  基线 topk6/drop1/hold1（当前生产参数）
  A*) 最小持有期 hold_thresh ∈ {5,10,20}  —— 治“排名抖动引起的来回切换”
  B*) 组合分散 topk ∈ {8,10}（n_drop=1）  —— 单仓变小，每次换仓占比下降
  C*) 信号平滑 score=近N日均值 ∈ {3,5}    —— 降低排名抖动本身
  D*) 组合：持有期+平滑 / 分散+持有期
每个变体在 5bp 与 10bp 滑点下各跑一次：既看当前成本假设下的净收益，
也检验悲观成本假设下是否存活（backtest-expert：策略要在悲观假设下存活）。

注意：本实验在同一OOS窗口上选参数，存在选择偏差；
采纳标准 = 出现“平原”（相邻参数同向改善），而非孤立尖点。

用法：
    python -m src.turnover
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

import config
from src.stress import _metrics


def _bt(signal: pd.Series, slippage: float, topk: int, n_drop: int,
        hold_thresh: int) -> pd.DataFrame:
    """与 train 同参回测（滑点与 TopkDropout 参数可调），返回完整日度报告。"""
    from src.utils import patch_qlib_deterministic
    patch_qlib_deterministic()  # 固定持仓列表顺序，保证回测跨进程可复现
    dates = signal.index.get_level_values("datetime").unique().sort_values()
    strategy = TopkDropoutStrategy(
        signal=signal, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh,
    )
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


def _smooth(pred: pd.Series, win: int) -> pd.Series:
    """打分按标的做 win 日均值平滑，降低排名抖动（仅用过去打分，无未来函数）。"""
    s = pred
    if s.index.names[0] != "instrument":
        s = s.reorder_levels(["instrument", "datetime"])
    s = s.sort_index()  # 标的内按时间排序，保证 rolling 方向正确
    return s.groupby(level="instrument", group_keys=False).apply(
        lambda x: x.rolling(win, min_periods=1).mean()
    )


# (名称, 策略参数, 平滑窗口|None)
VARIANTS = [
    ("V0 基线 k6/d1/h1",      dict(topk=6,  n_drop=1, hold_thresh=1),  None),
    ("A1 持有>=5日",          dict(topk=6,  n_drop=1, hold_thresh=5),  None),
    ("A2 持有>=10日",         dict(topk=6,  n_drop=1, hold_thresh=10), None),
    ("A3 持有>=20日",         dict(topk=6,  n_drop=1, hold_thresh=20), None),
    ("B1 k8/d1/h1",           dict(topk=8,  n_drop=1, hold_thresh=1),  None),
    ("B2 k10/d1/h1",          dict(topk=10, n_drop=1, hold_thresh=1),  None),
    ("C1 平滑3日",            dict(topk=6,  n_drop=1, hold_thresh=1),  3),
    ("C2 平滑5日",            dict(topk=6,  n_drop=1, hold_thresh=1),  5),
    ("D1 持有10日+平滑3日",   dict(topk=6,  n_drop=1, hold_thresh=10), 3),
    ("D2 k8+持有5日",         dict(topk=8,  n_drop=1, hold_thresh=5),  None),
]


def _row(rep: pd.DataFrame) -> dict:
    r = rep["return"] - rep["cost"]
    m = _metrics(r)
    to = rep["turnover"].fillna(0.0)
    cost = rep["cost"].fillna(0.0)
    return {**m, "日换手": to.mean(), "年成本": cost.mean() * 252.0}


def _fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("累计", "年化", "最大回撤"):
        out[c] = out[c].map(lambda x: f"{x:+.1%}")
    out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}")
    out["日换手"] = out["日换手"].map(lambda x: f"{x:.1%}")
    out["年成本"] = out["年成本"].map(lambda x: f"{x:.1%}")
    return out.to_string()


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    from src.overlay import _drop_qdii

    raw = _drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl"))

    tables: dict[float, dict[str, dict]] = {0.0005: {}, 0.0010: {}}
    for name, params, smooth_win in VARIANTS:
        sig = _smooth(raw, smooth_win) if smooth_win else raw
        for slip in tables:
            rep = _bt(sig, slip, **params)
            tables[slip][name] = _row(rep)
        print(f"[done] {name}", flush=True)

    for slip, rows in tables.items():
        print(f"\n===== 降换手对照 @ 滑点{slip*1e4:.0f}bp（无过滤信号，walk-forward样本外）=====")
        df = pd.DataFrame(rows).T[["累计", "年化", "Sharpe", "最大回撤", "日换手", "年成本"]]
        print(_fmt(df))


if __name__ == "__main__":
    main()
