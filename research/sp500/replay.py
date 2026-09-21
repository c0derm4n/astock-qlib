"""重放已冻结的入选策略；校验输入指纹，不修改正式发布指针。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from research.sp500.explore import BASE, frozen_market, signal_matrix, run_candidate
from research.sp500.validate import relative_metrics, ledger_audit


def replay(path=None):
    path = Path(path or Path(__file__).with_name("selected_strategy.json"))
    selected = json.loads(path.read_text(encoding="utf-8"))
    frozen = json.loads((BASE/selected["validation_directory"]/"frozen_validation.json").read_text(encoding="utf-8"))
    registry = json.loads((BASE/"output/sp500_exploration/runs"/frozen["protocol"]["exploration_run"]/"registry.json").read_text(encoding="utf-8"))
    if (selected["candidate"] != frozen["candidate"] or selected["start"] != registry["start"]
            or selected["end"] != registry["end"] or selected["comparison"] != registry["comparison"]
            or any(value != frozen["execution_policy"].get(key) for key, value in selected["policy"].items())):
        raise ValueError("策略配置与冻结验收规则不一致，请登记新的开发实验")
    for relative, expected in selected["required_hashes"].items():
        actual = hashlib.sha256((BASE/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"冻结研究输入已变化：{relative}；需显式重新研究，不能沿用本次结论")
    source = BASE / selected["source_directory"]
    market = frozen_market(source)
    scores = signal_matrix(market, selected["candidate"], None)
    report, trades, issues = run_candidate(market, scores, selected["candidate"], selected["policy"], selected["start"])
    summary, _ = relative_metrics(report, BASE / selected["benchmark_directory"], selected["comparison"])
    summary["ledger"] = ledger_audit(market, report, trades, selected["policy"], selected["policy"]["initial_capital"])
    summary["issues"] = issues
    output = BASE / "output/sp500_exploration/replays" / pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    report.to_csv(output / "report.csv")
    trades.to_csv(output / "trades.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy")
    args = parser.parse_args()
    replay(args.strategy)
