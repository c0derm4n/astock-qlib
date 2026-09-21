"""导入有来源的历史上市/清盘目录；目录本身不等于完整性审计通过。"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import config


def read_lifecycle(path=None):
    path = Path(path or config.DATA_DIR / "instrument_lifecycle.csv")
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "list_date", "delist_date", "source"])
    frame = pd.read_csv(path, dtype={"symbol": str})
    required = {"symbol", "list_date", "delist_date", "source"}
    if not required.issubset(frame.columns) or frame["symbol"].duplicated().any():
        raise ValueError("目录必须包含唯一symbol、list_date、delist_date、source")
    if not frame["symbol"].str.fullmatch(r"(?:SH5|SZ1)\d{5}").all():
        raise ValueError("目录只接受沪深ETF代码")
    if frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("历史目录必须逐行注明来源")
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="raise")
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="raise")
    if frame["list_date"].isna().any() or (frame["delist_date"] <= frame["list_date"]).any():
        raise ValueError("上市/清盘日期无效")
    return frame


def merge_lifecycle(meta, lifecycle):
    if lifecycle.empty:
        return meta.copy()
    out = meta.set_index("symbol").reindex(meta["symbol"].tolist() + [s for s in lifecycle["symbol"] if s not in set(meta["symbol"])])
    catalog = lifecycle.set_index("symbol")
    out.loc[catalog.index, "list_date"] = catalog["list_date"]
    out["delist_date"] = catalog["delist_date"].reindex(out.index)
    out["catalog_source"] = catalog["source"].reindex(out.index)
    return out.reset_index().rename(columns={"index": "symbol"})


def main():
    parser = argparse.ArgumentParser(description="校验历史标的目录；不自动宣称幸存者偏差已消除")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    frame = read_lifecycle(args.path)
    print(f"目录校验通过：{len(frame)}只，{frame['delist_date'].notna().sum()}只已清盘。")
    print("核对后将目录放入 data/instrument_lifecycle.csv，并补齐这些标的的历史行情，再重新研究。")


if __name__ == "__main__":
    main()
