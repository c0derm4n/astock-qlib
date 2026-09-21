"""不可变运行产物与单一发布指针，避免跨模型/配置混用结果。"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import subprocess
import uuid
from pathlib import Path
import pandas as pd
import config
from backend.artifacts import write_json_atomic


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol():
    path = config.BASE_DIR / "research_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["execution"] != "signal_close_t_execute_open_t_plus_1":
        raise ValueError("仅支持收盘信号、次日开盘执行")
    for key in ("initial_capital", "vol_target", "max_weight", "max_group_weight", "max_daily_turnover", "max_adv_participation"):
        if not math.isfinite(float(protocol[key])) or not 0 < float(protocol[key]):
            raise ValueError(f"研究参数 {key} 无效")
    if protocol["primary_model"] not in ("ridge", "lightgbm") or not {"momentum", protocol["primary_model"]}.issubset(protocol["candidate_models"]):
        raise ValueError("候选必须包含简单动量和支持的生产主模型")
    for key in ("stress_slippage_multipliers", "stress_capital_multipliers", "stress_fill_fractions", "stress_execution_delays"):
        values = protocol[key]
        if not values or len(set(values)) != len(values) or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"压力协议 {key} 无效")
    if 2 not in protocol["stress_slippage_multipliers"] or any(v > 1 for v in protocol["stress_fill_fractions"]):
        raise ValueError("须包含双倍滑点，成交比例不能超过1")
    if any(v != int(v) for v in protocol["stress_execution_delays"]) or not 0 < protocol["min_forward_fill_ratio"] <= 1:
        raise ValueError("执行延迟或前瞻成交比例无效")
    if protocol.get("portfolio_style") == "core_satellite":
        core = protocol["defensive_weights"]
        if (set(core) - {"SH511260", "SH518880"} or any(not 0 < v <= protocol["max_weight"] for v in core.values())
                or not 0 < protocol["satellite_weight"] <= 1
                or sum(core.values()) + protocol["satellite_weight"] > protocol["max_gross_weight"] + 1e-9):
            raise ValueError("底仓与轮动预算无效")
    if protocol.get("rank_buffer", 0) < 0:
        raise ValueError("排名缓冲不能为负")
    return protocol


def code_fingerprint():
    paths = [config.BASE_DIR / "config.py", config.BASE_DIR / "universe.py", config.BASE_DIR / "backend" / "artifacts.py"]
    paths += sorted((config.BASE_DIR / "src").rglob("*.py"))
    hashes = {str(p.relative_to(config.BASE_DIR)).replace("\\", "/"): sha256(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR, capture_output=True, text=True)
    return {"fingerprint": digest, "files": hashes, "git_head": git.stdout.strip() if git.returncode == 0 else None}


def new_run(protocol):
    run_id = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = config.OUTPUT_DIR / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "protocol.json", protocol)
    return run_id, directory


def artifact_path(name, legacy=None):
    pointer = config.OUTPUT_DIR / "current_run.json"
    if not pointer.exists():
        return Path(legacy) if legacy is not None else config.OUTPUT_DIR / name
    try:
        current = json.loads(pointer.read_text(encoding="utf-8"))
        directory = (config.OUTPUT_DIR / "runs" / current["run_id"]).resolve()
        if not directory.is_relative_to((config.OUTPUT_DIR / "runs").resolve()):
            raise ValueError("运行目录越界")
        return directory / name
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("当前运行指针损坏，停止读取混合产物") from exc


def publish(run_id, directory, manifest):
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "run_meta.json")
    manifest["artifacts"] = {p.relative_to(directory).as_posix(): sha256(p) for p in files}
    write_json_atomic(directory / "run_meta.json", manifest)
    write_json_atomic(config.OUTPUT_DIR / "current_run.json", {"run_id": run_id, "manifest_sha256": sha256(directory / "run_meta.json")})


def load_production():
    manifest_path = artifact_path("run_meta.json")
    pointer = config.OUTPUT_DIR / "current_run.json"
    if not pointer.exists():
        raise ValueError("尚无审计后的生产模型，请运行 python -m src.train")
    current = json.loads(pointer.read_text(encoding="utf-8"))
    if sha256(manifest_path) != current["manifest_sha256"]:
        raise ValueError("运行清单校验失败")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent / "production_model.pkl"
    if sha256(path) != manifest["artifacts"][path.name]:
        raise ValueError("模型内容校验失败")
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    if (bundle.get("run_id") != current["run_id"] or manifest.get("run_id") != current["run_id"]
            or bundle.get("protocol_hash") != manifest["protocol_hash"]
            or bundle.get("code_fingerprint") != manifest["code"]["fingerprint"]):
        raise ValueError("模型与运行清单版本不一致")
    return bundle, manifest
