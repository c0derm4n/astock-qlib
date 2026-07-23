"""共用工具：股票名称映射、代码转换、选股清单格式化。"""
from __future__ import annotations

import pandas as pd

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


def picks_table(score: pd.Series, names: dict[str, str], topk: int) -> pd.DataFrame:
    """把某一天的打分排序，取前 topk，生成带名称的清单表。"""
    s = score.sort_values(ascending=False).head(topk)
    rows = []
    for rank, (sym, sc) in enumerate(s.items(), 1):
        code = symbol_to_code(str(sym))
        rows.append(
            {
                "排名": rank,
                "代码": sym,
                "名称": names.get(code, ""),
                "打分": round(float(sc), 4),
            }
        )
    return pd.DataFrame(rows)
