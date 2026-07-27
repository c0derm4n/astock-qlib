"""数据血缘报告（Phase 3）：汇总最近一次拉取(ingest_log)与校验(validation_log)。

用法:
    python -m src.data_report              # 最近一次 run_id
    python -m src.data_report --run 20260723-153000
    python -m src.data_report --all        # 汇总全部历史
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from src.datasource import DataStore


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False) if not df.empty else "(空)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="指定 run_id，默认最近一次")
    parser.add_argument("--all", action="store_true", help="汇总全部历史 run")
    args = parser.parse_args()

    with DataStore() as store:
        run_id = None if args.all else (args.run or store.latest_run_id())
        if not args.all and run_id is None:
            raise SystemExit("暂无拉取记录，请先运行：python -m src.fetch_data")
        where = "" if args.all else f"WHERE run_id = '{run_id}'"
        scope = "全部历史" if args.all else run_id
        print(f"===== 数据血缘报告（{scope}）=====")

        # 1) 拉取汇总（按源/接口）
        ingest = store.con.execute(f"""
            SELECT source, interface,
                   count(*) AS n_symbols,
                   sum(rows) AS total_rows,
                   sum(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS n_ok,
                   sum(CASE WHEN status='fail' THEN 1 ELSE 0 END) AS n_fail,
                   min(date_min) AS date_min, max(date_max) AS date_max
            FROM ingest_log {where}
            GROUP BY source, interface ORDER BY source, interface
        """).df()
        print("\n[拉取 ingest_log]")
        print(_fmt(ingest))

        # 2) 失败明细（若有）
        fails = store.con.execute(f"""
            SELECT source, interface, symbol, error FROM ingest_log
            {('WHERE' if not where else where + ' AND')} status='fail'
            LIMIT 50
        """).df()
        if not fails.empty:
            print("\n[失败明细 (最多50条)]")
            print(_fmt(fails))

        # 3) 校验汇总（按类型）
        vwhere = "" if args.all else f"WHERE run_id = '{run_id}'"
        valid = store.con.execute(f"""
            SELECT check_type,
                   count(*) AS n_symbols,
                   sum(CASE WHEN passed THEN 1 ELSE 0 END) AS n_passed,
                   sum(n_mismatch) AS total_mismatch,
                   max(max_rel_diff) AS max_diff
            FROM validation_log {vwhere}
            GROUP BY check_type ORDER BY check_type
        """).df()
        print("\n[校验 validation_log]")
        print(_fmt(valid))
        if not valid.empty:
            print("说明：adj_consistency 同时验证了后复权/分红处理正确性"
                  "（后复权日收益 ≈ AKShare hfq 日收益）。")

        # 4) 导出扁平明细
        out = config.OUTPUT_DIR / "data_lineage.csv"
        detail = store.con.execute(f"""
            SELECT run_id, source, interface, symbol, rows, date_min, date_max,
                   status, error, fetch_time
            FROM ingest_log {where} ORDER BY fetch_time, source, interface, symbol
        """).df()
        detail.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n血缘明细已导出：{out}")


if __name__ == "__main__":
    main()
