"""对照图：多策略累计净值曲线（walk-forward 样本外）。

5 条线（同一次训练的信号，口径一致、含滑点）：
  ① 策略(趋势过滤TopK, 扣费+滑点)   —— output/predictions.pkl
  ② 策略(关闭趋势过滤)             —— output/predictions_raw.pkl
  ③ 等权+绝对动量择时(规则化基线)   —— 等权基准过去 W 日动量>0 满仓、否则空仓
  ④ 全ETF等权基准                  —— Qlib BENCH
  ⑤ 沪深300                        —— 回测基准 BENCH300

先跑 `python -m src.train`（会生成 predictions.pkl 与 predictions_raw.pkl），再：
    python -m src.plot_compare
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

import config


def _cum(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def _bt_daily(signal: pd.Series):
    """对给定信号跑 TopK 轮动回测（与 train 同参，含滑点）。
    返回 (策略日净收益 return-cost, 沪深300日收益 bench)。"""
    dates = signal.index.get_level_values("datetime").unique().sort_values()
    bt_start = dates[0]
    bt_end = dates[-2] if len(dates) >= 2 else dates[-1]
    slip = float(getattr(config, "SLIPPAGE_BPS", 0.0))
    strategy = TopkDropoutStrategy(signal=signal, topk=config.TOPK, n_drop=config.N_DROP)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    pd_dict, _ = backtest(
        start_time=bt_start, end_time=bt_end, strategy=strategy, executor=executor,
        account=100_000_000, benchmark=config.BENCHMARK_SYMBOL,
        exchange_kwargs={
            "freq": "day", "limit_threshold": config.LIMIT_THRESHOLD,
            "deal_price": getattr(config, "DEAL_PRICE", "close"),
            "open_cost": config.OPEN_COST + slip,
            "close_cost": config.CLOSE_COST + slip, "min_cost": config.MIN_COST,
        },
    )
    rep = list(pd_dict.values())[0][0]
    return (rep["return"] - rep["cost"]), rep["bench"]


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    filt = pd.read_pickle(config.OUTPUT_DIR / "predictions.pkl")
    raw = pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl")

    r_filt, csi = _bt_daily(filt)          # ① 趋势过滤策略 + 沪深300
    r_raw, _ = _bt_daily(raw)              # ② 关闭趋势过滤策略
    idx = r_filt.index

    # 等权基准：多取前 ~半年，保证 idx 起点动量已定义（避免年初空窗）
    pad = idx[0] - pd.Timedelta(days=220)
    cb = D.features([config.EQW_BENCH_SYMBOL], ["$close"], start_time=pad,
                    end_time=idx[-1], freq="day")["$close"].reset_index(
        level="instrument", drop=True).sort_index()
    W = int(getattr(config, "TREND_WINDOW", 60))
    mom = cb / cb.shift(W) - 1.0
    pos = (mom > 0).astype(float).shift(1).reindex(idx).fillna(0.0)  # 前一日决定，防未来函数
    eqw = cb.pct_change().reindex(idx)     # ④ 全ETF等权日收益
    r_timed = pos * eqw                    # ③ 等权+动量择时

    lines = [
        ("策略(趋势过滤TopK,扣费+滑点)", _cum(r_filt), "#d62728", 2.2),
        ("策略(关闭趋势过滤)", _cum(r_raw), "#ff7f0e", 1.6),
        ("等权+绝对动量择时(规则基线)", _cum(r_timed), "#9467bd", 2.0),
        ("全ETF等权基准", _cum(eqw), "#2ca02c", 1.6),
        ("沪深300", _cum(csi), "#1f77b4", 1.6),
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.5, 7))
    for name, cum, color, lw in lines:
        ax.plot(cum.index, cum.values * 100, label=f"{name}  {cum.iloc[-1]:+.1%}",
                color=color, linewidth=lw)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(f"多策略对照 · walk-forward样本外累计收益（{idx[0].date()} ~ {idx[-1].date()}）")
    ax.set_ylabel("累计收益率 (%)")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "equity_curve_compare.png"
    fig.savefig(out, dpi=130)
    print(f"已保存对照图：{out}")
    for name, cum, _c, _w in lines:
        print(f"  {name}: {cum.iloc[-1]:+.2%}")


if __name__ == "__main__":
    main()
