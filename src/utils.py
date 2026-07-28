"""共用工具：股票名称映射、代码转换、选股清单格式化、趋势过滤状态机。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import universe


def load_names() -> dict[str, str]:
    """返回 代码->名称 映射(6位代码)，来自内置 ETF 池。"""
    return dict(universe.UNIVERSE)


def symbol_to_code(symbol: str) -> str:
    """SH510300 -> 510300。"""
    return symbol[2:] if symbol[:2] in ("SH", "SZ", "BJ") else symbol


def to_qlib_symbol(code: str) -> str | None:
    """6位 ETF 代码 -> Qlib 符号：沪市(5开头)->SH，深市(1开头)->SZ；其余 None。"""
    code = str(code).zfill(6)
    if code[0] == "5":
        return "SH" + code
    if code[0] == "1":
        return "SZ" + code
    return None


def to_ts_code(code: str) -> str | None:
    """6位 ETF 代码 -> Tushare 码：510300 -> 510300.SH，159915 -> 159915.SZ。"""
    code = str(code).zfill(6)
    if code[0] == "5":
        return code + ".SH"
    if code[0] == "1":
        return code + ".SZ"
    return None


def to_ak_code(symbol: str) -> str:
    """SH510300 / 510300.SH -> 510300（AKShare fund_etf_hist_em 用纯6位码）。"""
    s = str(symbol)
    if s[:2] in ("SH", "SZ", "BJ"):
        return s[2:]
    if "." in s:
        return s.split(".")[0]
    return s.zfill(6)


def picks_table(score: pd.Series, names: dict[str, str], topk: int,
                premium: dict[str, float] | None = None) -> pd.DataFrame:
    """把某一天的打分排序，取前 topk，生成带名称的清单表。

    premium：{Qlib符号 -> 溢价率}，传入时额外输出“溢价%”列（仅 QDII 有值）。
    """
    s = score.sort_values(ascending=False).head(topk)
    rows = []
    for rank, (sym, sc) in enumerate(s.items(), 1):
        code = symbol_to_code(str(sym))
        row = {
            "排名": rank,
            "代码": sym,
            "名称": names.get(code, ""),
            "打分": round(float(sc), 4),
        }
        if premium is not None:
            p = premium.get(str(sym))
            row["溢价%"] = round(float(p) * 100, 2) if p is not None else ""
        rows.append(row)
    return pd.DataFrame(rows)


def drop_qdii(pred: pd.Series) -> pd.Series:
    """从信号中剔除 QDII 标的（与 config.EXCLUDE_QDII 生产口径一致）。"""
    insts = pred.index.get_level_values("instrument")
    bad = insts.map(lambda s: universe.get_asset_class(symbol_to_code(str(s))) == "qdii")
    return pred[~pd.Series(bad, index=pred.index)]


def patch_qlib_deterministic() -> None:
    """固定 qlib 回测的非确定性来源（幂等， qlib.init 后、backtest 前调用）。

    qlib.backtest.position.Position.get_stock_list 用 list(set(...)) 去重，
    set 迭代序受 PYTHONHASHSEED 影响、跨进程随机；该列表是 TopkDropoutStrategy
    卖出候选的输入顺序，quicksort 不稳定 → 分数接近的股票谁先卖随机，
    回测路径跨进程不可复现（同一信号两次回测累计收益可差 10+ 个百分点）。
    改为排序后的确定性顺序（只固定 tie-breaking，不改变策略逻辑）。
    """
    from qlib.backtest.position import Position

    def _get_stock_list_sorted(self):
        return sorted(set(self.position.keys())
                      - {"cash", "now_account_value", "cash_delay"})

    Position.get_stock_list = _get_stock_list_sorted


def trend_down_state(insts: list[str], start_time, end_time) -> pd.DataFrame:
    """趋势过滤 v2 状态机：返回 (datetime × instrument) 的 bool 表，
    True = 该日该标的判定为下行（应被保序降权）。仅用过去价格，无未来函数。

    规则（逐窗口 w ∈ TREND_WINDOWS）：
      mom_w = close/Ref(close,w)-1；
      迟滞带：mom < -TH → 下行(1)，mom > +TH → 上行(0)，之间沿用前一日状态；
      尚无穿越记录时退化为符号判定(mom<=0)；动量缺失(未上市)不罚。
    汇总：≥ TREND_VOTE_MIN 个窗口处于下行状态 → True；
    豁免：债/金等防守资产(TREND_EXEMPT_CLASSES)恒为 False。

    状态机从 start_time 前 ~550 自然日开始预热重放，保证首日状态已收敛；
    回测(train)与每日决策(decide)共用本函数，口径一致。需先 qlib.init。
    """
    from qlib.data import D  # 延迟导入：需在 qlib.init 之后调用

    windows = list(getattr(config, "TREND_WINDOWS", [config.TREND_WINDOW]))
    th = float(getattr(config, "TREND_HYSTERESIS", 0.0))
    vote_min = int(getattr(config, "TREND_VOTE_MIN", (len(windows) + 1) // 2))
    exempt = set(getattr(config, "TREND_EXEMPT_CLASSES", ()))

    pad = pd.Timestamp(start_time) - pd.Timedelta(days=550)
    fields = [f"$close/Ref($close,{w})-1" for w in windows]
    feat = D.features(sorted(set(insts)), fields, start_time=pad, end_time=end_time, freq="day")

    votes: pd.DataFrame | None = None
    for i in range(len(windows)):
        mom = feat.iloc[:, i].unstack("instrument").sort_index()  # datetime × instrument
        sig = pd.DataFrame(np.where(mom < -th, 1.0, np.where(mom > th, 0.0, np.nan)),
                           index=mom.index, columns=mom.columns)
        state = sig.ffill()
        fallback = ((mom <= 0) & mom.notna()).astype(float)  # 未穿越过阈值→符号判定
        state = state.where(state.notna(), fallback)
        votes = state if votes is None else votes + state
    down = votes >= vote_min
    for col in down.columns:  # 防守资产豁免
        if universe.get_asset_class(symbol_to_code(str(col))) in exempt:
            down[col] = False
    return down.loc[pd.Timestamp(start_time):]
