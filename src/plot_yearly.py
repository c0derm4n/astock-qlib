"""逐年收益柱状图（近十年扩展窗口 2018~2026）。

读 output/predictions_raw_long.pkl（src.train_long 生成），与生产同口径回测
（无过滤 TopK + config.HOLD_THRESH + 剔 QDII + 确定性 patch），
画 策略 / 沪深300 / 全ETF等权 逐年分组柱状图并打印对照表。

用法：
    python -m src.plot_yearly
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

import config
from src.plot_compare import _bt_daily, _metrics, _yearly
from src.utils import drop_qdii, patch_qlib_deterministic


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    patch_qlib_deterministic()

    signal = drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw_long.pkl"))
    r_strat, csi = _bt_daily(signal)
    idx = r_strat.index

    cb = D.features([config.EQW_BENCH_SYMBOL], ["$close"], start_time=idx[0],
                    end_time=idx[-1], freq="day")["$close"].reset_index(
        level="instrument", drop=True).sort_index()
    eqw = cb.pct_change().reindex(idx)

    hN = int(getattr(config, "HOLD_THRESH", 1))
    series = {
        f"当前策略(持有≥{hN}日)": (r_strat, "#d62728"),
        "沪深300": (csi, "#1f77b4"),
        "全ETF等权": (eqw, "#2ca02c"),
    }
    yearly = pd.DataFrame({name: _yearly(r) for name, (r, _c) in series.items()})

    # ---- 打印对照表（含超额列）----
    tab = yearly.copy()
    tab["超额(vs沪深300)"] = yearly.iloc[:, 0] - yearly["沪深300"]
    fmt = tab.map(lambda x: f"{x:+.1%}")
    print(f"===== 逐年收益（{int(yearly.index.min())}~{int(yearly.index.max())}，"
          f"样本外，扣费+滑点）=====")
    print(fmt.to_string())
    win = int((tab["超额(vs沪深300)"] > 0).sum())
    print(f"\n策略 vs 沪深300：{win}/{len(tab)} 年跑赢；"
          f"累计 {_metrics(r_strat)['累计']:+.1%} vs {_metrics(csi)['累计']:+.1%}")

    # ---- 分组柱状图 ----
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    years = yearly.index.astype(str)
    x = np.arange(len(years))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (name, (_r, color)) in enumerate(series.items()):
        v = yearly[name].to_numpy() * 100
        bars = ax.bar(x + (i - 1) * w, v, w, label=name, color=color, alpha=0.9)
        for b, val in zip(bars, v):
            ax.annotate(f"{val:+.0f}", (b.get_x() + b.get_width() / 2, val),
                        ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=8.5, color=color, fontweight="bold",
                        xytext=(0, 2 if val >= 0 else -2), textcoords="offset points")
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("年度收益率 (%)")
    ax.set_title(f"逐年收益对照 · walk-forward样本外（{years[0]}~{years[-1]}，扣费+滑点）")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "yearly_returns_long.png"
    fig.savefig(out, dpi=130)
    print(f"已保存逐年柱状图：{out}")


if __name__ == "__main__":
    main()
