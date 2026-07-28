"""近十年扩展窗口验证：walk-forward 测试年扩到 2018~2026。

数据自 2016-01 起，2018 是第一个可测试年（2016 训练 + 2017 验证）；
早年训练集薄、可交易 ETF 池小（多数行业 ETF 2019 年后上市）——
属真实历史约束，早期年份结果参考性弱，越近年份训练越充分。

不覆盖生产文件：信号存 output/predictions_raw_long.pkl，
图存 output/equity_curve_long.png；策略口径与生产一致
（无过滤 TopK + config.HOLD_THRESH + 剔 QDII + 确定性回测 patch）。

用法：
    python -m src.train_long
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

import config
from src.train import eval_ic, exclude_qdii_signal, walk_forward
from src.plot_compare import _bt_daily, _cum, _metrics, _yearly
from src.utils import patch_qlib_deterministic

TEST_YEARS = list(range(2018, 2027))


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    patch_qlib_deterministic()

    print(f"模式：近十年扩展 walk-forward（测试 {TEST_YEARS[0]}~{TEST_YEARS[-1]}）")
    pred, label, _model = walk_forward(TEST_YEARS)  # 不保存模型，避免覆盖生产 model.pkl
    pred.to_pickle(config.OUTPUT_DIR / "predictions_raw_long.pkl")

    print("\n===== 预测质量 IC（2018~2026 拼接样本外）=====")
    for k, v in eval_ic(pred, label).items():
        print(f"{k}: {v}")

    signal = exclude_qdii_signal(pred)  # 与生产同口径：剔 QDII（趋势过滤已关闭）
    r_strat, csi = _bt_daily(signal)    # _bt_daily 内已用 config.HOLD_THRESH
    idx = r_strat.index

    cb = D.features([config.EQW_BENCH_SYMBOL], ["$close"], start_time=idx[0],
                    end_time=idx[-1], freq="day")["$close"].reset_index(
        level="instrument", drop=True).sort_index()
    eqw = cb.pct_change().reindex(idx)

    hN = int(getattr(config, "HOLD_THRESH", 1))
    lines = [
        (f"当前策略(无过滤TopK·持有≥{hN}日)", _cum(r_strat), "#d62728", 2.4, r_strat),
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
    ax.set_title(f"近十年扩展验证 · walk-forward样本外累计收益"
                 f"（{idx[0].date()} ~ {idx[-1].date()}，扣费+滑点）")
    ax.set_ylabel("累计收益率 (%)")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "equity_curve_long.png"
    fig.savefig(out, dpi=130)
    print(f"\n已保存长窗口对照图：{out}")

    print("\n===== 指标对照（2018~2026 样本外，扣费+滑点）=====")
    tab = pd.DataFrame({name: _metrics(r) for name, _c, _clr, _w, r in lines}).T
    for c in ("累计", "年化", "最大回撤"):
        tab[c] = tab[c].map(lambda x: f"{x:+.1%}")
    tab["Sharpe"] = tab["Sharpe"].map(lambda x: f"{x:.2f}")
    print(tab[["累计", "年化", "Sharpe", "最大回撤"]].to_string())

    print("\n===== 逐年收益 =====")
    ytab = pd.DataFrame({name: _yearly(r) for name, _c, _clr, _w, r in lines})
    print(ytab.map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()
