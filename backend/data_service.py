"""读取 output/ 下的产物（预测、回测报告、训练指标、数据溯源），为 API 提供数据。

通过当前运行指针读取版本化产物，复用 src.utils 的工具函数。
预测/回测产物由 train 管道生成；按文件版本缓存，并复用预测日期索引。
"""
from __future__ import annotations

import json
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.artifacts import finite_float
from src.utils import load_names, picks_table
from src.runs import artifact_path

_cache: OrderedDict[str, tuple[tuple[int, ...], object]] = OrderedDict()
_cache_lock = threading.RLock()
ANNUALIZATION_DAYS = 252


def _artifact(path):
    try:
        return artifact_path(path.name, path)
    except ValueError as exc:
        raise ArtifactUnavailableError(str(exc)) from exc


class ArtifactUnavailableError(RuntimeError):
    """产物正在重写或尚不完整，调用方应稍后重试。"""


def _file_version(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


def _cached(path: Path, loader: Callable[[Path], object]):
    key = str(path)
    # FastAPI 同步接口在线程池中执行，冷启动并发查询只允许加载一次。
    with _cache_lock:
        try:
            version = _file_version(path)
            hit = _cache.get(key)
            if hit and hit[0] == version:
                _cache.move_to_end(key)
                return hit[1]
            obj = loader(path)
            if _file_version(path) != version:
                raise ArtifactUnavailableError(f"{path.name} 正在更新，请稍后重试")
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            raise ArtifactUnavailableError(f"{path.name} 暂时不可读取，请稍后重试") from exc
        _cache[key] = (version, obj)
        _cache.move_to_end(key)
        if len(_cache) > 32:
            _cache.popitem(last=False)
        return obj


# ---------------------------------------------------------------------------
# 数据溯源（train 经 src.runs.publish 写入版本化 run_meta.json）
# ---------------------------------------------------------------------------
_RUN_META_FILE = config.OUTPUT_DIR / "run_meta.json"


def read_run_meta() -> dict:
    """读数据版本与结构化指标；缺失/损坏返空 dict。"""
    path = _artifact(_RUN_META_FILE)
    if not path.exists():
        return {}
    try:
        meta = _cached(path,
                       lambda p: json.loads(p.read_text(encoding="utf-8")))
        return meta if isinstance(meta, dict) else {}
    except ArtifactUnavailableError:
        return {}


# ---------------------------------------------------------------------------
# 预测 / 选 ETF 清单（对应 src.predict 的逻辑）
# ---------------------------------------------------------------------------
_PRED_FILE = config.OUTPUT_DIR / "predictions.pkl"


def has_predictions() -> bool:
    return _artifact(_PRED_FILE).exists()


@dataclass(frozen=True)
class _Predictions:
    scores: pd.Series
    dates: pd.DatetimeIndex
    date_strings: tuple[str, ...]


def _load_predictions() -> _Predictions:
    def _read(path: Path) -> _Predictions:
        pred = pd.read_pickle(path)
        if isinstance(pred, pd.DataFrame) and pred.shape[1] == 1:
            pred = pred.iloc[:, 0]
        if not isinstance(pred, pd.Series) or not isinstance(pred.index, pd.MultiIndex):
            raise ValueError("预测数据必须为 (datetime, instrument) 索引的打分序列")
        if set(pred.index.names) != {"datetime", "instrument"}:
            raise ValueError("预测数据缺少 datetime/instrument 索引")
        pred = pd.to_numeric(pred, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        pred = pred.sort_index()
        dates = pred.index.get_level_values("datetime")
        if not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None or dates.hasnans:
            raise ValueError("预测日期必须为有效的无时区日期")
        if pred.index.has_duplicates:
            raise ValueError("预测数据包含重复日期/标的")
        dates = dates.unique().sort_values()
        return _Predictions(pred, dates, tuple(dates.strftime("%Y-%m-%d")))

    return _cached(_artifact(_PRED_FILE), _read)


def available_dates() -> list[str]:
    return list(_load_predictions().date_strings)


def get_picks(date: str | None, topk: int) -> dict:
    """与 src.predict 一致：取 target 当日打分 TopK；非交易日回退到之前最近交易日。"""
    if topk < 1 or topk > 50:
        raise ValueError("TopK 必须在 1 到 50 之间")
    predictions = _load_predictions()
    dates = predictions.dates
    if dates.empty:
        raise ValueError("预测文件没有有效打分，请重新运行训练")
    try:
        target = pd.Timestamp(date) if date else dates[-1]
    except (TypeError, ValueError) as exc:
        raise ValueError("日期必须为有效的 YYYY-MM-DD") from exc
    if pd.isna(target) or target.tzinfo is not None:
        raise ValueError("日期必须为有效的 YYYY-MM-DD，不包含时区")
    position = dates.searchsorted(target.normalize(), side="right") - 1
    if position < 0:
        raise ValueError(f"没有 {target.date()} 及之前的预测数据")
    target = dates[position]
    table = picks_table(predictions.scores.xs(target, level="datetime"), load_names(), topk)
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
# bench 为沪深300ETF对照 csi300_etf）
# ---------------------------------------------------------------------------
_REPORT_FILE = config.OUTPUT_DIR / "backtest_report.csv"


def has_backtest() -> bool:
    return _artifact(_REPORT_FILE).exists()


def _load_report() -> pd.DataFrame:
    def _read(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        required = ["return", "bench", "cost"]
        if not set(required).issubset(df.columns):
            raise ValueError("回测报告缺少 return/bench/cost 列")
        df.index = pd.to_datetime(df.index, errors="raise")
        if df.index.hasnans:
            raise ValueError("回测报告包含无效日期")
        df[required] = df[required].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return df.sort_index()

    return _cached(_artifact(_REPORT_FILE), _read)


def backtest_series() -> dict:
    """累计净值曲线（策略扣费后 vs 沪深300基准），供前端画图。"""
    df = _load_report()
    strat = df["return"] - df["cost"]
    cum_s = (1 + strat).cumprod() - 1
    cum_b = (1 + df["bench"]).cumprod() - 1
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "strategy": [finite_float(x, 6) for x in cum_s],
        "benchmark": [finite_float(x, 6) for x in cum_b],
    }


def backtest_summary() -> dict:
    """回测汇总指标，采用v2固定研究口径。
    策略年化使用实际日期CAGR；日均超额按252日年化，回撤基于复利净值。"""
    df = _load_report()
    strat = df["return"] - df["cost"]
    excess = df["return"] - df["bench"] - df["cost"]
    n = len(df)
    cum_strat = float((1 + strat).prod(min_count=1) - 1)
    cum_bench = float((1 + df["bench"]).prod(min_count=1) - 1)
    mean, std = float(excess.mean()), float(excess.std(ddof=1))
    cs_strat = (1 + strat).cumprod()
    cs_excess = cs_strat / (1 + df["bench"]).cumprod()
    years = (df.index[-1] - df.index[0]).days / 365.25 if n > 1 else 0
    return {
        "start": df.index[0].strftime("%Y-%m-%d") if n else None,
        "end": df.index[-1].strftime("%Y-%m-%d") if n else None,
        "days": n,
        "annualization_days": ANNUALIZATION_DAYS,
        "strategy_cum_return": finite_float(cum_strat, 6),
        "bench_cum_return": finite_float(cum_bench, 6),
        "strategy_annual": finite_float((1 + cum_strat) ** (1 / years) - 1, 6) if years > 0 and cum_strat > -1 else None,
        "excess_annual": finite_float(mean * ANNUALIZATION_DAYS, 6),
        "information_ratio": finite_float(mean / std * np.sqrt(ANNUALIZATION_DAYS), 4)
        if np.isfinite(std) and std > 0 else None,
        "excess_max_drawdown": finite_float((cs_excess / cs_excess.cummax().clip(lower=1.) - 1).min(), 6),
        "strategy_max_drawdown": finite_float((cs_strat / cs_strat.cummax().clip(lower=1.) - 1).min(), 6),
        "metric_basis": "compound_nav",
    }


# ---------------------------------------------------------------------------
# 每日决策（src.decide 写出的 decision_YYYYMMDD.csv：操作/代码/名称/打分/排名/现价）
# ---------------------------------------------------------------------------
def _read_positions() -> dict:
    """当前份额/现金账本，只能通过成交回报和公司行动更新。"""
    from src.account import read_account
    try:
        obj = read_account()
        return {
            "positions": sorted(obj["holdings"]), "holdings": obj["holdings"],
            "cash": obj["cash"], "revision": obj["revision"],
            "updated_at": obj.get("updated_at"), "valid": True,
        }
    except ValueError as exc:
        return {"positions": [], "valid": False, "error": str(exc)}


def latest_decision() -> dict | None:
    latest = config.OUTPUT_DIR / "latest_decision.json"
    if latest.exists():
        decision = _cached(latest, lambda p: json.loads(p.read_text(encoding="utf-8")))
        names = load_names()
        rows = [{"操作": "买入" if o["side"] == "buy" else "卖出", "代码": o["symbol"],
                 "名称": names.get(o["symbol"][2:], ""), "份额": o["quantity"],
                 "参考收盘价": o["reference_price"], "目标权重": o["target_weight"],
                 "订单编号": o["order_id"]} for o in decision["orders"]]
        return {**decision, "date": decision["signal_date"], "rows": rows, "positions": _read_positions()}
    files = sorted(config.OUTPUT_DIR.glob("decision_*.csv"))
    if not files:
        return None
    f = files[-1]
    df = pd.read_csv(f).fillna("")
    return {
        "date": f.stem.replace("decision_", ""),
        "rows": df.to_dict("records"),
        "positions": _read_positions(),
        "legacy": True,
    }


# ---------------------------------------------------------------------------
# 总览（供 Dashboard 聚合展示）
# ---------------------------------------------------------------------------
def overview() -> dict:
    """口径与当前生产逻辑对齐：walk-forward 滚动重训 + 动态池/QDII 排除，
    可交易数以版本化 run_meta.json 为准（含动态池过滤结果）。"""
    names = load_names()
    meta = read_run_meta()
    protocol = meta.get("protocol", {})
    return {
        "universe_size": len(names),
        "n_tradable": meta.get("n_tradable"),
        "data_version": meta.get("data_version"),
        "calendar_end": meta.get("calendar_end"),
        "has_predictions": has_predictions(),
        "has_backtest": has_backtest(),
        "topk": protocol.get("topk", config.TOPK),
        "n_drop": protocol.get("n_drop", config.N_DROP),
        "hold_thresh": protocol.get("hold_days", config.HOLD_THRESH),
        "label_horizon": protocol.get("label_horizon", config.LABEL_HORIZON),
        "walk_forward": config.WALK_FORWARD,
        "wf_train_start": config.WF_TRAIN_START,
        "wf_test_years": list(config.WF_TEST_YEARS),
        "exclude_qdii": config.EXCLUDE_QDII,
        "use_dynamic_universe": config.USE_DYNAMIC_UNIVERSE,
        "investment_gate": meta.get("investment_gate"),
        "production": meta.get("production"),
        "primary_model": protocol.get("primary_model"),
        "portfolio_style": protocol.get("portfolio_style"),
        "defensive_weights": protocol.get("defensive_weights", {}),
    }


def read_acceptance() -> dict | None:
    path = _artifact(config.OUTPUT_DIR / "acceptance.json")
    if not path.exists():
        return None
    report = _cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))
    if not isinstance(report, dict) or not isinstance(report.get("gate"), dict):
        raise ArtifactUnavailableError("验收报告结构损坏，停止展示准入状态")
    return report
