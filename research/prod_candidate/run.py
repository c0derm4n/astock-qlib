"""生产线候选（src/ core_satellite）在 2018-2025 的独立样本外回测。

与 src/research.py 的区别：
- 读本目录自己的 protocol.json（样本外起点前移到 2018），不读也不写 research_protocol.json
- 产物只写入 output/prod_candidate/，不调用 src.runs.publish，因此不切换 output/current_run.json
- 策略结构与执行引擎完全复用 src.market / src.models / src.simulation，口径与生产一致

窗口下界的由来（见 src/models.py:60-68）：walk-forward 每个测试年需要"测试年前一年作验证集、
更早的样本作训练集"，且训练样本 <200 或验证样本 <50 时直接抛错。行情数据自 2016-01-04 起，
故 2016/2017 作测试年时训练样本为 0，2018（训练 515 行 / 验证 1128 行）是最早可行起点。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.artifacts import write_json_atomic
from src.market import load_market
from src.models import walk_forward_models
from src.simulation import performance, simulate

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PROTOCOL = HERE / "protocol.json"
PARENT_PROTOCOL = BASE / "research_protocol.json"
OUT = BASE / "output/prod_candidate"

_POSITIVE_KEYS = ("initial_capital", "vol_target", "max_weight", "max_group_weight",
                  "max_daily_turnover", "max_adv_participation", "satellite_weight")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_local_protocol() -> dict:
    """读取并校验本目录协议。规则对齐 src/runs.read_protocol，但指向本地文件。"""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["execution"] != "signal_close_t_execute_open_t_plus_1":
        raise ValueError("仅支持收盘信号、次日开盘执行")
    for key in _POSITIVE_KEYS:
        value = float(protocol[key])
        if not 0 < value:
            raise ValueError(f"研究参数 {key} 无效")
    if protocol["primary_model"] not in ("ridge", "lightgbm") or not {
            "momentum", protocol["primary_model"]}.issubset(protocol["candidate_models"]):
        raise ValueError("候选必须包含简单动量和支持的生产主模型")
    for key in ("stress_slippage_multipliers", "stress_capital_multipliers",
                "stress_fill_fractions", "stress_execution_delays"):
        values = protocol[key]
        if not values or len(set(values)) != len(values) or any(v <= 0 for v in values):
            raise ValueError(f"压力协议 {key} 无效")
    if 2 not in protocol["stress_slippage_multipliers"]:
        raise ValueError("须包含双倍滑点")
    core = protocol.get("defensive_weights") or {}
    if protocol.get("portfolio_style") == "core_satellite":
        if set(core) - {"SH511260", "SH518880"} or any(not 0 < v <= protocol["max_weight"] for v in core.values()):
            raise ValueError("底仓与轮动预算无效")
        if sum(core.values()) + protocol["satellite_weight"] > protocol["max_gross_weight"] + 1e-9:
            raise ValueError("底仓与轮动预算超出总仓上限")
    if protocol.get("rank_buffer", 0) < 0:
        raise ValueError("排名缓冲不能为负")
    if pd.Timestamp(protocol["research_start"]) >= pd.Timestamp(protocol["research_end"]):
        raise ValueError("样本外起止无效")
    return protocol


def run():
    protocol = read_local_protocol()
    market = load_market(horizon=protocol["label_horizon"])
    end = str(market.calendar[-1].date())
    if end != protocol["research_end"]:
        raise ValueError(f"行情截止日 {end} 与协议登记的 {protocol['research_end']} 不一致；更新协议后再运行")

    run_id = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = OUT / "runs" / run_id
    (directory / "research_source").mkdir(parents=True, exist_ok=False)
    for path in sorted(HERE.glob("*.py")):
        shutil.copyfile(path, directory / "research_source" / path.name)
    write_json_atomic(directory / "protocol.json", protocol)
    write_json_atomic(directory / "manifest.json", {
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protocol_sha256": sha256(PROTOCOL), "parent_protocol_sha256": sha256(PARENT_PROTOCOL),
        "production_pointer_unchanged": True, "research_only": True,
        "data_end": end, "trading_symbols": sorted(market.bars.symbol.unique()),
        "engine": {str(p.relative_to(BASE)).replace("\\", "/"): sha256(p) for p in (
            BASE / "src/market.py", BASE / "src/models.py",
            BASE / "src/portfolio.py", BASE / "src/simulation.py")}})

    forecasts, folds = walk_forward_models(market, protocol)
    write_json_atomic(directory / "folds.json", {"folds": folds})
    summaries = {}
    for name, scores in forecasts.items():
        report, trades, issues = simulate(market, scores, protocol)
        summaries[name] = performance(report, protocol["annualization_days"])
        summaries[name]["execution_limits"] = issues
        summaries[name]["trade_count"] = int(len(trades))
        summaries[name]["average_exposure"] = float((report.value / report.account).mean())
        report.to_csv(directory / f"report_{name}.csv")
        trades.to_csv(directory / f"trades_{name}.csv", index=False)
        scores.to_pickle(directory / f"predictions_{name}.pkl")
        print(f"{name}: 累计 {summaries[name]['cumulative_return']:.2%} "
              f"年化 {summaries[name]['cagr']:.2%} 回撤 {summaries[name]['max_drawdown']:.2%}", flush=True)

    write_json_atomic(directory / "results.json", {
        "schema_version": 1, "run_id": run_id, "research_only": True,
        "window": {"start": protocol["research_start"], "end": protocol["research_end"],
                   "fold_years": [f["year"] for f in folds]},
        "primary_model": protocol["primary_model"], "portfolio_style": protocol["portfolio_style"],
        "summaries": summaries, "protocol_sha256": sha256(PROTOCOL),
        "parent_protocol_sha256": sha256(PARENT_PROTOCOL),
        "production_pointer_unchanged": True,
        "first_fold_rows": folds[0]["train_rows"] if folds else None})
    write_json_atomic(directory / "completion.json", {
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "models": sorted(summaries), "folds": len(folds), "production_pointer_unchanged": True})
    write_json_atomic(OUT / "runs" / "latest.json", {"directory": str(directory)})
    print(f"产物: {directory}")
    return directory


if __name__ == "__main__":
    run()
