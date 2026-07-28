"""读取 output/ 下的产物（预测、回测报告、训练指标），为 API 提供数据。

只读文件 + 复用 src.utils 的工具函数，不改动 src 内任何代码。
预测/回测产物由 train 管道生成；文件较多按 mtime 做简单缓存。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.utils import load_names, picks_table

_cache: dict[str, tuple[float, object]] = {}


def _cached(path: Path, loader: Callable[[Path], object]):
    key = str(path)
    mtime = path.stat().st_mtime
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    obj = loader(path)
    _cache[key] = (mtime, obj)
    return obj


# ---------------------------------------------------------------------------
# 预测 / 选 ETF 清单（对应 src.predict 的逻辑）
# ---------------------------------------------------------------------------
_PRED_FILE = config.OUTPUT_DIR / "predictions.pkl"


def has_predictions() -> bool:
    return _PRED_FILE.exists()


def _load_pred() -> pd.Series:
    return _cached(_PRED_FILE, pd.read_pickle)  # Series: (datetime, instrument) -> score


def available_dates() -> list[str]:
    pred = _load_pred()
    dates = pred.index.get_level_values("datetime").unique().sort_values()
    return [d.strftime("%Y-%m-%d") for d in dates]


def get_picks(date: str | None, topk: int) -> dict:
    """与 src.predict 一致：取 target 当日打分 TopK；非交易日回退到之前最近交易日。"""
    pred = _load_pred()
    dates = pred.index.get_level_values("datetime")
    target = pd.Timestamp(date) if date else dates.max()
    if target not in set(dates):
        earlier = dates[dates <= target]
        if len(earlier) == 0:
            raise ValueError(f"没有 {target.date()} 及之前的预测数据")
        target = earlier.max()
    table = picks_table(pred.xs(target, level="datetime"), load_names(), topk)
    rows = [
        {
            "rank": int(r["排名"]),
            "code": str(r["代码"]),
            "name": r["名称"],
            "score": float(r["打分"]),
        }
        for r in table.to_dict("records")
    ]
    return {"date": target.strftime("%Y-%m-%d"), "topk": topk, "rows": rows}


# ---------------------------------------------------------------------------
# 回测报告（train 写出的 backtest_report.csv，含 return/bench/cost 等列；
# bench 为沪深300代理 BENCH300）
# ---------------------------------------------------------------------------
_REPORT_FILE = config.OUTPUT_DIR / "backtest_report.csv"


def has_backtest() -> bool:
    return _REPORT_FILE.exists()


def _load_report() -> pd.DataFrame:
    def _read(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df.sort_index()

    return _cached(_REPORT_FILE, _read)


def backtest_series() -> dict:
    """累计净值曲线（策略扣费后 vs 沪深300基准），供前端画图。"""
    df = _load_report()
    strat = df["return"] - df["cost"]
    cum_s = (1 + strat).cumprod() - 1
    cum_b = (1 + df["bench"]).cumprod() - 1
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "strategy": [round(float(x), 6) for x in cum_s],
        "benchmark": [round(float(x), 6) for x in cum_b],
    }


def backtest_summary() -> dict:
    """回测汇总指标；超额序列的统计口径与 qlib risk_analysis 保持一致。"""
    df = _load_report()
    strat = df["return"] - df["cost"]
    excess = df["return"] - df["bench"] - df["cost"]
    n = len(df)
    cum_strat = float((1 + strat).prod() - 1)
    cum_bench = float((1 + df["bench"]).prod() - 1)
    mean, std = float(excess.mean()), float(excess.std(ddof=1))
    cs = excess.cumsum()
    return {
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
        "days": n,
        "strategy_cum_return": round(cum_strat, 6),
        "bench_cum_return": round(cum_bench, 6),
        "strategy_annual": round((1 + cum_strat) ** (252 / n) - 1, 6) if n else None,
        "excess_annual": round((1 + mean) ** 252 - 1, 6),
        "information_ratio": round(mean / std * np.sqrt(252), 4) if std else None,
        "max_drawdown": round(float((cs - cs.cummax()).min()), 6),
    }


# ---------------------------------------------------------------------------
# 每日决策（src.decide 写出的 decision_YYYYMMDD.csv：操作/代码/名称/打分/排名/现价）
# ---------------------------------------------------------------------------
def latest_decision() -> dict | None:
    files = sorted(config.OUTPUT_DIR.glob("decision_*.csv"))
    if not files:
        return None
    f = files[-1]
    df = pd.read_csv(f).fillna("")
    return {
        "date": f.stem.replace("decision_", ""),
        "rows": df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# 总览（供 Dashboard 聚合展示）
# ---------------------------------------------------------------------------
def overview() -> dict:
    names = load_names()
    return {
        "universe_size": len(names),
        "has_predictions": has_predictions(),
        "has_backtest": has_backtest(),
        "topk": config.TOPK,
        "label_horizon": config.LABEL_HORIZON,
        "train_period": list(config.TRAIN_PERIOD),
        "valid_period": list(config.VALID_PERIOD),
        "test_period": list(config.TEST_PERIOD),
    }
