"""每日盘中决策：交易日 14:30 运行，判定今天要买入/卖出哪些 ETF。

只操作 ETF，且不考虑 QDII（config.EXCLUDE_QDII，QDII 彻底不参与；历史遗留的
QDII 持仓会被建议清仓）。用 14:30 的实时价近似当日收盘价，复现回测同款规则：

流程（对齐回测口径，无未来函数）：
  1. 增量补齐日线到 DuckDB（昨日及之前的正式K线）；
  2. 拉取实时快照(东财 fund_etf_spot_em)，拼出"今日临时K线"（现价≈收盘价）；
  3. 重写 Qlib 数据（复用 dump_qlib.run_dump：动态池 + QDII 排除）；
  4. 加载 models/model.pkl 对今日打分（Alpha158，与训练同口径归一化）；
  5. 叠加趋势过滤（过去 TREND_WINDOW 日动量<=0 降权）；
  6. 与 output/positions.json 持仓对比，按 TopkDropout 同款规则给出
     买入/卖出/持有清单（每日最多换 N_DROP 只；空仓则一次建仓 TopK）。

注意：快照拼的"今日K线"只写进 Qlib 数据(盘中临时)，不落 DuckDB；盘中增量若拉到
今日部分K线，次日增量会从最后一天(含)续拉并用收盘正式数据覆盖，自愈无残留。

用法:
    python -m src.decide                # 完整流程：补数 + 快照 + 打分 + 买卖判定
    python -m src.decide --apply        # 判定后把结果写回 positions.json（确认执行）
    python -m src.decide --no-refresh   # 跳过增量补数（数据已是最新时更快）
    python -m src.decide --force        # 忽略交易日/时间检查（复盘调试用）

定时：用 Windows 任务计划在交易日 14:30 调用 run_decide_1430.ps1（见 README）。
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import argparse
import json
import pickle

import numpy as np
import pandas as pd

import config
import universe
from src.datasource import DataStore
from src import dump_qlib
from src import fetch_data
from src.utils import load_names, symbol_to_code, to_qlib_symbol


# --------------------------------------------------------------------------
# 1) 数据：增量补数 + 今日盘中快照
# --------------------------------------------------------------------------
def refresh_history(store: DataStore) -> None:
    """增量补齐正式日线（昨日及之前）到 DuckDB，复用 fetch_data 的拉取逻辑。"""
    codes = fetch_data.get_universe()
    run_id = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S") + "-decide"
    fetch_data.ingest(store, codes, run_id, force=False)


_SPOT_RENAME = {
    "代码": "code", "名称": "name", "最新价": "last", "开盘价": "open",
    "最高价": "high", "最低价": "low", "昨收": "prev_close",
    "成交量": "volume", "成交额": "amount",
}
_SINA_RENAME = {
    "代码": "code", "名称": "name", "最新价": "last", "今开": "open",
    "最高": "high", "最低": "low", "昨收": "prev_close",
    "成交量": "volume", "成交额": "amount",
}


def _spot_em() -> pd.DataFrame:
    """东财实时快照（成交量单位：手）。"""
    import akshare as ak

    df = ak.fund_etf_spot_em().rename(columns=_SPOT_RENAME)
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def _spot_sina() -> pd.DataFrame:
    """新浪实时快照兑底（东财不可达时；成交量 股->手 对齐东财口径）。"""
    import akshare as ak

    df = ak.fund_etf_category_sina(symbol="ETF基金").rename(columns=_SINA_RENAME)
    df["code"] = df["code"].astype(str).str[-6:]  # sh510300 -> 510300
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100.0
    return df


def fetch_spot() -> pd.DataFrame:
    """实时快照（东财优先，不可达降级新浪）-> 以 Qlib 符号为索引的
    {last,open,high,low,prev_close,volume,amount}。

    只保留 ETF 池内、且当日有成交(last>0)的标的。两源都失败时抛异常。"""
    try:
        df = _spot_em()
    except Exception as e:
        print(f"  [降级] 东财快照不可达({str(e)[:60]})，改用新浪实时源")
        df = _spot_sina()
    need = [c for c in _SPOT_RENAME.values() if c in df.columns]
    df = df[need].copy()
    codes = set(universe.get_codes())
    df = df[df["code"].isin(codes)]
    df["symbol"] = df["code"].map(lambda c: to_qlib_symbol(c))
    for c in ("last", "open", "high", "low", "prev_close", "volume", "amount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["last"].notna() & (df["last"] > 0)]
    return df.set_index("symbol")


def append_today_bars(data: dict[str, pd.DataFrame], raw_last_close: dict[str, float],
                      spot: pd.DataFrame, today: str) -> int:
    """把 14:30 快照拼成"今日临时后复权K线"追加到各标的日线尾部。

    hfq 价 = 现价原价 × 最近复权因子（因子取昨日 hfq_close/raw_close，盘中分红
    调整未知，近似可接受）。已有今日行(盘中增量拉到的部分K线)则整行替换。
    """
    n = 0
    for sym, df in data.items():
        if sym.startswith("BENCH") or sym not in spot.index:
            continue
        row = spot.loc[sym]
        prev_raw = raw_last_close.get(sym)
        if not prev_raw or prev_raw <= 0 or df.empty:
            continue
        hist = df[df["date"] < today]
        if hist.empty:
            continue
        factor = float(hist["close"].iloc[-1]) / prev_raw  # 昨日 hfq/raw 反推复权因子
        o = float(row.get("open") or row["last"]) * factor
        h = float(row.get("high") or row["last"]) * factor
        l = float(row.get("low") or row["last"]) * factor
        c = float(row["last"]) * factor
        prev_c = float(row.get("prev_close") or prev_raw)
        new_row = {
            "date": today, "open": o, "high": h, "low": l, "close": c,
            "volume": float(row.get("volume") or 0.0),
            "vwap": (h + l + c) / 3.0, "factor": 1.0,
            "change": float(row["last"]) / prev_c - 1.0 if prev_c > 0 else 0.0,
        }
        data[sym] = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        n += 1
    return n


# --------------------------------------------------------------------------
# 2) 打分：加载已训模型，对今日做 Alpha158 推断 + 趋势过滤
# --------------------------------------------------------------------------
def score_today(today: str) -> pd.Series:
    """返回今日各可交易 ETF 的策略打分（已含趋势过滤；QDII 已被 dump 排除）。"""
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D
    from qlib.data.dataset import DatasetH

    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)

    model_file = config.MODEL_DIR / "model.pkl"
    if not model_file.exists():
        raise SystemExit("未找到 models/model.pkl，请先运行：python -m src.train")
    stale_days = (pd.Timestamp.now() - pd.Timestamp(model_file.stat().st_mtime, unit="s")).days
    if stale_days > int(getattr(config, "MODEL_STALE_DAYS", 45)):
        print(f"提示：模型已 {stale_days} 天未重训（>{config.MODEL_STALE_DAYS}天），建议重跑 python -m src.train")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # 归一化拟合区间与最后一次 walk-forward 训练保持一致（防口径漂移）
    if getattr(config, "WALK_FORWARD", False):
        ty = config.WF_TEST_YEARS[-1]
        fit = (config.WF_TRAIN_START, f"{ty - 2}-12-31")
    else:
        fit = config.TRAIN_PERIOD
    h = config.LABEL_HORIZON
    label = ([f"Ref($close, -{h + 1})/Ref($close, -1) - 1"], ["LABEL0"])
    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    handler = Alpha158(
        instruments=config.MARKET,
        start_time=fit[0], end_time=today,
        fit_start_time=fit[0], fit_end_time=fit[1],
        infer_processors=infer_processors, label=label,
    )
    # 只推断最近一小段（含今日），避免整段预测浪费时间
    seg_start = (pd.Timestamp(today) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    ds = DatasetH(handler, segments={"test": (seg_start, today)})
    pred = model.predict(ds, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    dates = pred.index.get_level_values("datetime")
    if pd.Timestamp(today) not in set(dates):
        raise SystemExit(f"打分结果中没有 {today}（今日K线未生成？检查快照/补数是否成功）")
    day = pred.xs(pd.Timestamp(today), level="datetime").astype(float)

    # 趋势过滤（同 train.trend_filtered_signal 口径）：过去 N 日动量<=0 → 降权
    if getattr(config, "USE_TREND_FILTER", False):
        w = config.TREND_WINDOW
        insts = sorted(day.index)
        feat = D.features(insts, [f"$close/Ref($close,{w})-1"],
                          start_time=today, end_time=today, freq="day")
        mom = feat.iloc[:, 0].droplevel("datetime").reindex(day.index)
        penalty = float(np.nanmin(day.to_numpy())) - 1000.0
        down = (mom <= 0).to_numpy(dtype=bool)
        day.iloc[down] = penalty
        print(f"趋势过滤：{int(down.sum())}/{len(day)} 只因过去{w}日动量<=0被降权")

    # 保险：QDII 一律不参与（正常已被 dump 从 etf.txt 排除）
    qdii = [s for s in day.index if universe.get_asset_class(symbol_to_code(str(s))) == "qdii"]
    if qdii:
        day = day.drop(qdii)
    return day.sort_values(ascending=False)


# --------------------------------------------------------------------------
# 3) 判定：与持仓对比，按 TopkDropout 同款规则给出买卖清单
# --------------------------------------------------------------------------
def load_positions() -> list[str]:
    f = config.POSITIONS_FILE
    if not f.exists():
        return []
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        return [str(s) for s in obj.get("positions", [])]
    except Exception:
        return []


def save_positions(symbols: list[str]) -> None:
    config.POSITIONS_FILE.write_text(json.dumps({
        "positions": sorted(symbols),
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def make_decision(held: list[str], score: pd.Series,
                  topk: int, n_drop: int) -> tuple[list[str], list[str], list[str]]:
    """返回 (买入, 卖出, 持有)。规则与回测 TopkDropoutStrategy 对齐：

    - 空仓：一次建仓打分最高的 TopK；
    - 有持仓：跌出 TopK 的持仓按打分从差到好最多卖 N_DROP 只，用最高分的
      未持仓标的补足到 TopK（控制换手）；
    - 持仓中的 QDII 一律建议清仓（不占 N_DROP 名额，策略性退出）；
    - 持仓当日无打分（停牌/被动态池剔除）：保持并提示人工处理。
    """
    s = score.dropna()
    ranked = list(s.sort_values(ascending=False).index)
    rank_of = {sym: i + 1 for i, sym in enumerate(ranked)}

    # 策略性清仓：QDII 不再操作
    forced_sells = [h for h in held
                    if universe.get_asset_class(symbol_to_code(h)) == "qdii"]
    held = [h for h in held if h not in forced_sells]

    if not held:  # 空仓（或仅剩 QDII）：一次建仓
        buys = ranked[:topk]
        return buys, forced_sells, []

    held_known = [h for h in held if h in rank_of]
    held_unknown = [h for h in held if h not in rank_of]  # 无打分：保持+人工确认
    top = set(ranked[:topk])
    keep = [h for h in held_known if h in top]
    out = sorted((h for h in held_known if h not in top),
                 key=lambda x: rank_of[x], reverse=True)  # 排名最差的先卖
    sells = out[:n_drop]
    stay = keep + [h for h in out if h not in sells] + held_unknown
    n_buy = max(0, topk - len(stay))
    buys = [x for x in ranked if x not in set(held)][:n_buy]
    return buys, forced_sells + sells, stay


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="把判定结果写回 positions.json")
    parser.add_argument("--no-refresh", action="store_true", help="跳过增量补数")
    parser.add_argument("--force", action="store_true", help="忽略交易日/时间检查")
    parser.add_argument("--topk", type=int, default=config.TOPK)
    args = parser.parse_args()

    now = pd.Timestamp.now()
    today = now.strftime("%Y-%m-%d")
    if not args.force:
        if now.weekday() >= 5:
            raise SystemExit(f"{today} 是周末非交易日，跳过（--force 可强制运行）")
        if now.strftime("%H:%M") < "14:00":
            print(f"提示：建议 {config.DECISION_RUN_TIME} 运行（现在 {now:%H:%M}，"
                  f"现价代替收盘价的误差会更大）")

    # ---- 数据：补数 + 今日快照 + 重写 Qlib ----
    with DataStore() as store:
        if not args.no_refresh:
            print("== 第1步：增量补齐正式日线 ==")
            refresh_history(store)
        print("\n== 第2步：拉取盘中实时快照，拼今日临时K线 ==")
        try:
            spot = fetch_spot()
        except Exception as e:
            raise SystemExit(f"实时快照拉取失败（东财接口不可达？稍后重试）：{e}")
        if spot.empty:
            raise SystemExit(f"快照无池内 ETF 成交数据，{today} 可能是非交易日（--force 可跳过检查）")

        data = dump_qlib.load_all(store)
        if not data:
            raise SystemExit("DuckDB 无数据，请先运行：python -m src.fetch_data")
        raw = store.load_bars_raw()
        raw = raw[raw["date"] < today]
        raw_last_close = {str(s): float(g["close"].iloc[-1])
                          for s, g in raw.groupby("symbol") if not g.empty}
        n = append_today_bars(data, raw_last_close, spot, today)
        print(f"今日临时K线：{n} 只（现价≈收盘价）")
        if n == 0:
            raise SystemExit("没有任何标的拼出今日K线，无法决策")

        print("\n== 第3步：重写 Qlib 数据（动态池 + 排除QDII）==")
        dump_qlib.run_dump(store, data)

    # ---- 打分 + 买卖判定 ----
    print("\n== 第4步：模型打分（Alpha158 + 趋势过滤）==")
    score = score_today(today)

    held = load_positions()
    buys, sells, stays = make_decision(held, score, args.topk, config.N_DROP)

    # ---- 输出 ----
    names = load_names()
    rank_of = {sym: i + 1 for i, sym in enumerate(score.index)}

    def _rows(symbols: list[str], action: str) -> list[dict]:
        out = []
        for sym in symbols:
            code = symbol_to_code(str(sym))
            last = spot.loc[sym, "last"] if sym in spot.index else None
            out.append({
                "操作": action, "代码": sym, "名称": names.get(code, ""),
                "打分": round(float(score[sym]), 4) if sym in score.index else "",
                "排名": rank_of.get(sym, ""),
                "现价": round(float(last), 3) if last is not None else "",
            })
        return out

    rows = _rows(sells, "卖出") + _rows(buys, "买入") + _rows(stays, "持有")
    table = pd.DataFrame(rows)
    out_file = config.OUTPUT_DIR / f"decision_{now:%Y%m%d}.csv"
    table.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"\n===== {today} {now:%H:%M} 盘中决策（Top{args.topk}，每日最多换 {config.N_DROP} 只，QDII 不参与）=====")
    if held:
        print(f"当前持仓({len(held)})：{', '.join(held)}")
    else:
        print("当前持仓：空仓 → 按 TopK 一次建仓")
    print(table.to_string(index=False) if not table.empty else "（无操作）")
    if not buys and not sells:
        print("结论：今日无需买卖，继续持有。")
    print(f"\n清单已保存：{out_file}")

    if args.apply:
        new_pos = sorted(set(stays) | set(buys))
        save_positions(new_pos)
        print(f"已更新持仓文件：{config.POSITIONS_FILE}（{len(new_pos)} 只）")
    else:
        print("提示：确认按清单执行后，运行 python -m src.decide --no-refresh --apply 更新持仓；"
              "或手工编辑 output/positions.json。")
    print("提示：这是模型决策支持，不是投资建议；14:30 现价≈收盘价存在尾盘偏差。")


if __name__ == "__main__":
    main()
