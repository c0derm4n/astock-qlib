"""波动率目标仓位层（B方向研究）：风控从"标的级趋势过滤"挪到"组合级仓位调节"。

思路（Barroso & Santa-Clara 2015）：模型信号原样用（保留选股能力），
每日按 权益仓位 w = min(1, 目标年化波动 / 组合近 N 日实现波动) 缩放敞口，
其余 (1-w) 停泊国债ETF；w 用昨日及之前收益计算（shift(1)，无未来函数），
仓位变化超过缓冲带才调仓，每单位换手扣双边成本。

对照（同一次 walk-forward 训练的样本外信号，无需重训）：
  ① 无过滤 TopK（基线）            —— output/predictions_raw.pkl
  ② ①+波动率目标（12%/15%/18% 扫描找平原）
  ③ 趋势过滤v2 TopK                —— output/predictions.pkl
  ④ 全ETF等权基准 / ⑤ 沪深300

先跑 `python -m src.train`（生成两个 pkl），再：
    python -m src.overlay
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

import config


def vol_target_overlay(r: pd.Series, r_park: pd.Series,
                       target_ann: float | None = None,
                       lookback: int | None = None,
                       band: float | None = None) -> tuple[pd.Series, pd.Series]:
    """对日收益序列 r 叠加波动率目标仓位层，返回 (叠加后日收益, 权益仓位序列)。

    w_t = clip(目标年化波动 / 昨日已知的 N 日实现年化波动, 0, 1)；
    |w 变化| > band 才实际调仓；每单位换手扣 (买+卖佣金+双边滑点) 成本；
    (1-w) 部分获得停泊资产收益 r_park。仅用过去数据，无未来函数。
    """
    target = float(target_ann if target_ann is not None else config.VOL_TARGET_ANNUAL)
    lb = int(lookback if lookback is not None else config.VOL_LOOKBACK)
    bd = float(band if band is not None else config.VOL_REBAL_BAND)
    slip = float(getattr(config, "SLIPPAGE_BPS", 0.0))
    cost_unit = config.OPEN_COST + config.CLOSE_COST + 2 * slip  # 卖权益+买停泊 双腿

    sigma = r.rolling(lb).std() * np.sqrt(252.0)
    w_raw = (target / sigma).clip(upper=1.0).shift(1)  # 昨日已知波动 → 今日仓位

    w_list, cur = [], 1.0  # 期初满仓
    for x in w_raw.to_numpy():
        if np.isfinite(x) and abs(x - cur) > bd:
            cur = float(x)
        w_list.append(cur)
    w = pd.Series(w_list, index=r.index, name="equity_weight")

    turnover = w.diff().abs().fillna(0.0)
    park = r_park.reindex(r.index).fillna(0.0)
    ret = w * r + (1.0 - w) * park - turnover * cost_unit
    return ret, w


def _metrics(r: pd.Series) -> dict:
    """年化收益/波动、Sharpe、最大回撤、累计收益。"""
    r = r.fillna(0.0)
    nav = (1.0 + r).cumprod()
    ann = nav.iloc[-1] ** (252.0 / len(r)) - 1.0
    vol = r.std() * np.sqrt(252.0)
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"累计": nav.iloc[-1] - 1.0, "年化": ann, "年化波动": vol,
            "Sharpe": ann / vol if vol > 0 else float("nan"), "最大回撤": mdd}


def _yearly(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).groupby(r.index.year).prod() - 1.0


def _drop_qdii(pred: pd.Series) -> pd.Series:
    """兼容别名：实现已下沉到 src.utils.drop_qdii。"""
    from src.utils import drop_qdii
    return drop_qdii(pred)


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)
    from src.plot_compare import _bt_daily  # 复用同参回测（含成本滑点）

    raw = _drop_qdii(pd.read_pickle(config.OUTPUT_DIR / "predictions_raw.pkl"))
    filt = pd.read_pickle(config.OUTPUT_DIR / "predictions.pkl")

    r_raw, csi = _bt_daily(raw)    # ① 无过滤基线 + 沪深300
    r_filt, _ = _bt_daily(filt)    # ③ 趋势过滤v2
    idx = r_raw.index

    park = D.features([config.VOL_PARK_SYMBOL], ["$close"], start_time=idx[0],
                      end_time=idx[-1], freq="day")["$close"].reset_index(
        level="instrument", drop=True).sort_index().pct_change().reindex(idx)
    eqw = D.features([config.EQW_BENCH_SYMBOL], ["$close"], start_time=idx[0],
                     end_time=idx[-1], freq="day")["$close"].reset_index(
        level="instrument", drop=True).sort_index().pct_change().reindex(idx)

    variants: dict[str, pd.Series] = {"无过滤TopK(基线)": r_raw}
    weights: dict[str, pd.Series] = {}
    for tgt in (0.12, 0.15, 0.18):  # 目标波动扫描：找平原不找尖峰
        ret, w = vol_target_overlay(r_raw, park, target_ann=tgt)
        name = f"基线+波动率目标{tgt:.0%}"
        variants[name] = ret
        weights[name] = w
    variants["趋势过滤v2 TopK"] = r_filt
    variants["全ETF等权"] = eqw
    variants["沪深300"] = csi

    print("\n===== 波动率目标仓位层 · 对照（walk-forward 样本外，扣费+滑点）=====")
    rows = {k: _metrics(v) for k, v in variants.items()}
    tab = pd.DataFrame(rows).T
    for c in ("累计", "年化", "年化波动", "最大回撤"):
        tab[c] = tab[c].map(lambda x: f"{x:+.1%}")
    tab["Sharpe"] = tab["Sharpe"].map(lambda x: f"{x:.2f}")
    print(tab.to_string())

    print("\n===== 逐年收益 =====")
    ytab = pd.DataFrame({k: _yearly(v) for k, v in variants.items()})
    print(ytab.map(lambda x: f"{x:+.1%}").to_string())
    ytab.to_csv(config.OUTPUT_DIR / "overlay_yearly.csv", encoding="utf-8-sig")

    # 图：上=净值曲线，下=权益仓位(15%目标)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, axw) = plt.subplots(2, 1, figsize=(12.5, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    styles = {
        "无过滤TopK(基线)": ("#ff7f0e", 1.6),
        "基线+波动率目标15%": ("#d62728", 2.2),
        "趋势过滤v2 TopK": ("#8c564b", 1.4),
        "全ETF等权": ("#2ca02c", 1.4),
        "沪深300": ("#1f77b4", 1.4),
    }
    for name, (color, lw) in styles.items():
        cum = (1.0 + variants[name].fillna(0.0)).cumprod() - 1.0
        ax.plot(cum.index, cum.values * 100, label=f"{name}  {cum.iloc[-1]:+.1%}",
                color=color, linewidth=lw)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title(f"波动率目标仓位层 · walk-forward样本外（{idx[0].date()} ~ {idx[-1].date()}）")
    ax.set_ylabel("累计收益率 (%)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)

    w15 = weights["基线+波动率目标15%"]
    axw.fill_between(w15.index, w15.values * 100, color="#d62728", alpha=0.35)
    axw.set_ylabel("权益仓位 (%)")
    axw.set_xlabel("日期")
    axw.set_ylim(0, 105)
    axw.grid(alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "equity_curve_overlay.png"
    fig.savefig(out, dpi=130)
    print(f"\n已保存对照图：{out}")


if __name__ == "__main__":
    main()
