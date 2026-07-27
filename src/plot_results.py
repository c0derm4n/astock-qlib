"""第 5 步（可选）：把回测结果画成累计净值曲线对比图。

读取 output/backtest_report.csv（策略日净收益=return−cost、沪深300=bench）
+ Qlib 里的 BENCH 实例（全ETF等权），画三条累计收益曲线，输出 PNG。

用法:
    python -m src.plot_results
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 无显示环境，仅存文件
import matplotlib.pyplot as plt
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

import config


def _cum(returns: pd.Series) -> pd.Series:
    """日收益 -> 累计收益（复利）。"""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def _eqw_returns(idx: pd.DatetimeIndex) -> pd.Series | None:
    """从 Qlib BENCH（全ETF等权）close 取日收益，对齐到回测日期。"""
    try:
        qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
        c = D.features([config.EQW_BENCH_SYMBOL], ["$close"],
                       start_time=idx[0], end_time=idx[-1], freq="day")["$close"]
        c = c.reset_index(level="instrument", drop=True).reindex(idx).ffill()
        return c.pct_change()
    except Exception as e:
        print("等权基准取数失败（跳过该线）:", e)
        return None


def main() -> None:
    rep = pd.read_csv(config.OUTPUT_DIR / "backtest_report.csv",
                      parse_dates=["datetime"]).set_index("datetime")
    idx = rep.index
    curves = {
        "策略(趋势过滤TopK, 扣费+滑点)": (_cum(rep["return"] - rep["cost"]), "#d62728", 2.0),
        "全ETF等权基准": (None, "#2ca02c", 1.7),
        "沪深300": (_cum(rep["bench"]), "#1f77b4", 1.7),
    }
    eqw = _eqw_returns(idx)
    if eqw is not None:
        curves["全ETF等权基准"] = (_cum(eqw), "#2ca02c", 1.7)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, (cum, color, lw) in curves.items():
        if cum is None:
            continue
        ax.plot(cum.index, cum.values * 100, label=f"{name}  {cum.iloc[-1]:+.1%}",
                color=color, linewidth=lw)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(f"ETF轮动策略 vs 基准 · walk-forward样本外累计收益"
                 f"（{idx[0].date()} ~ {idx[-1].date()}）")
    ax.set_ylabel("累计收益率 (%)")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "equity_curve.png"
    fig.savefig(out, dpi=130)
    print(f"已保存曲线图：{out}")


if __name__ == "__main__":
    main()
