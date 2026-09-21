"""先登记候选，再按同一成本与两个固定起点回放；全部结果留存。"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from research.ex_nasdaq.core import market30, execute, validate_candidate, SYMBOLS
from research.sp500.validate import relative_metrics

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "output/sp500_ex_nasdaq"
_MARKET = None


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize(source):
    global _MARKET
    _MARKET = market30(source)


def evaluate(candidate, registry, directory):
    began = time.monotonic()
    if str(_MARKET.calendar[-1].date()) != registry['end']:
        raise ValueError('冻结行情截止日与登记不一致')
    policy = json.loads((BASE / registry["source"] / "protocol.json").read_text(encoding="utf-8"))
    policy.update(registry["execution"])
    policy.update(candidate.get("policy_overrides", {}))
    results = {}
    for label, start in registry["starts"].items():
        report, trades, issues = execute(_MARKET, candidate, policy, start)
        metrics, benchmark = relative_metrics(report, BASE / registry["benchmark"], registry["comparison"])
        metrics.update({"issues": issues, "trades": len(trades),
                        "traded_symbols": sorted(trades.symbol.unique()) if not trades.empty else [],
                        "average_exposure": float((report.value/report.account).mean())})
        report.to_csv(Path(directory) / f"report_{candidate['name']}_{label}.csv")
        trades.to_csv(Path(directory) / f"trades_{candidate['name']}_{label}.csv", index=False)
        benchmark.to_csv(Path(directory) / f"benchmark_{candidate['name']}_{label}.csv")
        results[label] = metrics
    results["seconds"] = time.monotonic() - began
    return candidate["name"], results


def run(registry_path, workers=3):
    path = Path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    for candidate in registry["candidates"]:
        validate_candidate(candidate)
    names = [c["name"] for c in registry["candidates"]]
    if len(set(names)) != len(names):
        raise ValueError("候选名称重复")
    source = BASE / registry["source"]
    directory = OUT / "runs" / (pd.Timestamp.now().strftime("%Y%m%d-%H%M%S") + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "registry.json", registry)
    files = list((BASE / "research/ex_nasdaq").glob("*.py"))
    files += list((BASE / "research/sp500").glob("*.py"))
    files += list((BASE / "src").glob("*.py"))
    files += [BASE / "config.py", BASE / "universe.py", path.resolve()]
    files += [source / n for n in ("market_snapshot.parquet", "nav_snapshot.parquet", "metadata_snapshot.parquet", "protocol.json", "acceptance.json")]
    files += [p for p in (BASE / registry["benchmark"]).iterdir() if p.is_file()]
    for candidate in registry['candidates']:
        if candidate.get('predictions'):
            files += [p for p in (BASE/candidate['predictions']).parent.iterdir() if p.is_file()]
    pointer = (BASE / "output/current_run.json").read_bytes()
    write_json(directory / "manifest.json", {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "symbols": sorted(SYMBOLS),
        "research_only": True, "hashes": {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    for file in files:
        if file.suffix == ".py":
            target = directory / "code_snapshot" / file.relative_to(BASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    results = {}
    print(f"运行目录：{directory}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(source,)) as pool:
        futures = {pool.submit(evaluate, c, registry, directory): c["name"] for c in registry["candidates"]}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            write_json(directory / "results.json", results)
            desc = "; ".join(f"{k}: 年化{v['cagr']:.2%} 累计{v['cumulative_return']:.2%} 回撤{v['max_drawdown']:.2%} 达标={v['target_met']}" for k,v in result.items() if k != "seconds")
            print(f"{name}: {desc}", flush=True)
    if pointer != (BASE / "output/current_run.json").read_bytes():
        raise ValueError("生产指针意外变化")
    write_json(directory / "completion.json", {"candidate_count": len(results), "production_pointer_unchanged": True})
    write_json(OUT / "runs/latest.json", {"directory": str(directory)})
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.registry, args.workers)
