"""对照图：多策略累计净值曲线（walk-forward 样本外，剔 QDII，与生产同口径）。

5 条线（同一次训练的信号，口径一致、含滑点；策略层 hold_thresh 取 config.HOLD_THRESH）：
  ① 当前策略(无过滤TopK·持有≥N日, 扣费+滑点) —— output/predictions_raw.pkl
  ② 已否决: 趋势过滤TopK                  —— output/predictions.pkl
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
from src.utils import drop_qdii


def _metrics(r: pd.Series) -> dict:
    nav = (1.0 + r.fillna(0.0)).cumprod()
    ann = nav.iloc[-1] ** (252.0 / len(r)) - 1.0
    vol = r.std() * (252.0 ** 0.5)
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"累计": nav.iloc[-1] - 1.0, "年化": ann,
            "Sharpe": ann / vol if vol > 0 else float("nan"), "最大回撤": mdd}


def _yearly(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).groupby(r.index.year).prod() - 1.0


def _cum(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def _bt_daily(signal: pd.Series):
    """对给定信号跑 TopK 轮动回测（与 train 同参，含滑点）。
    返回 (策略日净收益 return-cost, 沪深300日收益 bench)。"""
    from src.utils import patch_qlib_deterministic
    patch_qlib_deterministic()  # 固定持仓列表顺序，保证回测跨进程可复现
    dates = signal.index.get_level_values("datetime").unique().sort_values()
    bt_start = dates[0]
    bt_end = dates[-2] if len(dates) >= 2 else dates[-1]
    slip = float(getattr(config, "SLIPPAGE_BPS", 0.0))
    strategy = TopkDropoutStrategy(signal=signal, topk=config.TOPK, n_drop=config.N_DROP,
                                       hold_thresh=getattr(config, "HOLD_THRESH", 1))
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
    # 剔 QDII：与 config.EXCLUDE_QDII 生产口径一致（旧信号 pkl 中含纳指/恒生）
    filt = drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions.pkl"))
    raw = drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl"))

    r_raw, csi = _bt_daily(raw)            # ① 当前策略(无过滤+hN) + 沪深300
    r_filt, _ = _bt_daily(filt)            # ② 已否决: 趋势过滤
    idx = r_raw.index

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

    hN = int(getattr(config, "HOLD_THRESH", 1))
    lines = [
        (f"当前策略(无过滤TopK·持有≥{hN}日)", _cum(r_raw), "#d62728", 2.6, r_raw),
        ("已否决:趋势过滤TopK", _cum(r_filt), "#ff9896", 1.2, r_filt),
        ("等权+绝对动量择时(规则基线)", _cum(r_timed), "#9467bd", 1.6, r_timed),
        ("全ETF等权基准", _cum(eqw), "#2ca02c", 1.6, eqw),
        ("沪深300", _cum(csi), "#1f77b4", 1.8, csi),
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.5, 7))
    for name, cum, color, lw, _r in lines:
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

    # ---- 指标表 + 逐年收益 ----
    tab = pd.DataFrame({name: _metrics(r) for name, _c, _clr, _w, r in lines}).T
    for c in ("累计", "年化", "最大回撤"):
        tab[c] = tab[c].map(lambda x: f"{x:+.1%}")
    tab["Sharpe"] = tab["Sharpe"].map(lambda x: f"{x:.2f}")
    print("\n===== 指标对照（样本外，扣费+滑点）=====")
    print(tab[["累计", "年化", "Sharpe", "最大回撤"]].to_string())

    print("\n===== 逐年收益 =====")
    ytab = pd.DataFrame({name: _yearly(r) for name, _c, _clr, _w, r in lines})
    print(ytab.map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
