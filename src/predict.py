"""
第 4 步：输出选股清单（决策支持）。

读取 train.py 生成的预测结果，列出某个交易日打分最高的 TopK 只股票。
默认展示最新交易日；可用 --date 指定历史某日复盘。

用法:
    python -m src.predict                 # 最新交易日 TopK
    python -m src.predict --date 2025-06-30 --topk 20
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from src.utils import load_names, picks_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="交易日 YYYY-MM-DD，默认最新")
    parser.add_argument("--topk", type=int, default=config.TOPK, help="展示前几只")
    args = parser.parse_args()

    pred_file = config.OUTPUT_DIR / "predictions.pkl"
    if not pred_file.exists():
        raise SystemExit("未找到预测结果，请先运行：python -m src.train")
    pred = pd.read_pickle(pred_file)  # Series: (datetime, instrument) -> score

    dates = pred.index.get_level_values("datetime")
    target = pd.Timestamp(args.date) if args.date else dates.max()
    if target not in set(dates):  # 回退到不晚于目标的最近交易日
        earlier = dates[dates <= target]
        if len(earlier) == 0:
            raise SystemExit(f"没有 {target.date()} 及之前的预测数据")
        target = earlier.max()

    names = load_names()
    table = picks_table(pred.xs(target, level="datetime"), names, args.topk)
    out = config.OUTPUT_DIR / f"picks_{target.strftime('%Y%m%d')}.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"===== {target.date()}  模型打分 Top{args.topk} 选股清单 =====")
    print(table.to_string(index=False))
    print(f"\n已保存：{out}")
    print("提示：这是模型打分排名，不是投资建议；请结合基本面与风控，务必先模拟盘验证。")


if __name__ == "__main__":
    main()
