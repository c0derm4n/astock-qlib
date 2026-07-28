"""临时检查脚本：核对补数后的 DuckDB 覆盖情况。"""
import duckdb

con = duckdb.connect("data/market.duckdb", read_only=True)
print("== 各最新日期的标的数 ==")
print(con.execute(
    "SELECT d1, count(*) syms FROM (SELECT symbol, max(date) d1 FROM bars_raw GROUP BY symbol) GROUP BY d1 ORDER BY d1"
).df().to_string())
print("\n== 未到全局最新日期的标的 ==")
print(con.execute(
    "SELECT symbol, max(date) d1 FROM bars_raw GROUP BY symbol "
    "HAVING max(date) < (SELECT max(date) FROM bars_raw) ORDER BY symbol"
).df().to_string())
print("\n== 本次 run(20260727-195157) 接口状态 ==")
print(con.execute(
    "SELECT interface, status, count(*) n FROM ingest_log "
    "WHERE run_id='20260727-195157' GROUP BY interface, status ORDER BY interface, status"
).df().to_string())
print("\n== 失败明细 ==")
print(con.execute(
    "SELECT symbol, interface, error FROM ingest_log "
    "WHERE run_id='20260727-195157' AND status != 'ok'"
).df().to_string())
