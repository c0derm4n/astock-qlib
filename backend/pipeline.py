"""管道任务管理：以子进程方式运行 src 的四步命令，捕获日志与训练指标。

- 命令与 README 快速开始完全一致：python -m src.fetch_data / dump_qlib / train
- 同一时间只允许一个管道任务（训练会重写 qlib 数据与模型，不宜并发）
- 每步完整日志写入 output/pipeline_logs/<step>.log；train 成功后解析
  其 stdout 中的 IC / 回测指标，落盘 output/train_metrics.json 供 API 读取
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque

import config
from backend import data_service

BASE_DIR = config.BASE_DIR
LOG_DIR = config.OUTPUT_DIR / "pipeline_logs"
METRICS_FILE = config.OUTPUT_DIR / "train_metrics.json"

# step -> (中文名, 命令)
STEPS: dict[str, tuple[str, list[str]]] = {
    "fetch": ("抓取行情", [sys.executable, "-m", "src.fetch_data"]),
    "dump": ("转换Qlib数据", [sys.executable, "-m", "src.dump_qlib"]),
    "train": ("训练+回测", [sys.executable, "-m", "src.train"]),
    "decide": ("每日决策", [sys.executable, "-m", "src.decide"]),
}
STEP_ORDER = ["fetch", "dump", "train", "decide"]
# 「一键运行」默认只跑数据+训练三步；decide 为每日盘中决策，按需单独触发
DEFAULT_STEPS = ["fetch", "dump", "train"]


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.current_step: str | None = None
        self.step_status: dict[str, str] = {}  # step -> pending/running/done/failed
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.log: deque[str] = deque(maxlen=1000)  # 最近日志行(尾部)


_state = _State()


def _fmt_ts(ts: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _append(line: str) -> None:
    with _state.lock:
        _state.log.append(line.rstrip("\n"))


def _run_step(step: str) -> bool:
    """运行单个步骤，流式捕获输出；返回是否成功(exit code 0)。"""
    _, cmd = STEPS[step]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # 子进程强制 UTF-8 输出，避免 Windows 默认 GBK 解码乱码
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(LOG_DIR / f"{step}.log", "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _append(line)
            lf.write(line)
        return proc.wait() == 0


# ---------------------------------------------------------------------------
# train 输出指标解析
# ---------------------------------------------------------------------------
_RE_IC = re.compile(r"^(IC均值|ICIR|RankIC均值|RankICIR)\s*[:：]\s*(\S+)")
_RE_RANGE = re.compile(r"^区间[:：]\s*(\S+)\s*~\s*(\S+).*?持仓\s*(\d+)\s*只")
_RE_STRAT = re.compile(r"策略累计收益\(扣费\)[:：]\s*(-?[\d.]+%)")
_RE_BENCH300 = re.compile(r"沪深300累计[:：]\s*(-?[\d.]+%)")
_RE_EQW = re.compile(r"等权ETF基准累计[:：]\s*(-?[\d.]+%)")
_RE_EXCESS = re.compile(
    r"超额年化\(vs沪深300\)[:：]\s*(-?[\d.]+%)\s*信息比率IR[:：]\s*(-?[\d.na]+)\s*超额最大回撤[:：]\s*(-?[\d.]+%)"
)
_RE_STRAT_ABS = re.compile(r"策略自身[:：]\s*年化\s*(-?[\d.]+%)\s*最大回撤\s*(-?[\d.]+%)")


def _pct(s: str) -> float | None:
    try:
        return float(s.rstrip("%"))
    except ValueError:
        return None


def parse_train_metrics(text: str) -> dict:
    """从 train 的 stdout 文本中解析 IC 与回测指标（百分比数值化，单位 %）。"""
    ic: dict[str, float | None] = {}
    backtest: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if m := _RE_IC.match(line):
            try:
                ic[m.group(1)] = float(m.group(2))
            except ValueError:
                ic[m.group(1)] = None
        elif m := _RE_RANGE.match(line):
            backtest.update({"start": m.group(1), "end": m.group(2), "topk": int(m.group(3))})
        elif m := _RE_STRAT.search(line):
            backtest["strategy_cum_return_pct"] = _pct(m.group(1))
        elif m := _RE_BENCH300.search(line):
            backtest["hs300_cum_return_pct"] = _pct(m.group(1))
        elif m := _RE_EQW.search(line):
            backtest["eqw_cum_return_pct"] = _pct(m.group(1))
        elif m := _RE_EXCESS.search(line):
            backtest["excess_annual_pct"] = _pct(m.group(1))
            try:
                backtest["information_ratio"] = float(m.group(2))
            except ValueError:
                backtest["information_ratio"] = None
            backtest["excess_max_drawdown_pct"] = _pct(m.group(3))
        elif m := _RE_STRAT_ABS.search(line):
            backtest["strategy_annual_pct"] = _pct(m.group(1))
            backtest["strategy_max_drawdown_pct"] = _pct(m.group(2))
    return {"ic": ic, "backtest": backtest}


def _merge_run_meta(metrics: dict) -> None:
    """train 会把结构化 IC/回测指标写进 output/run_meta.json（权威源），
    用它覆盖正则解析结果，正则仅兜底并补充 run_meta 缺失的字段
    （等权ETF基准累计、策略年化、topk）。"""
    meta = data_service.read_run_meta()
    if isinstance(meta.get("ic_metrics"), dict):
        metrics["ic"].update(meta["ic_metrics"])
    bt = meta.get("backtest") or {}

    def _pct100(key: str) -> float | None:
        v = bt.get(key)
        return round(float(v) * 100, 2) if v is not None else None

    overlay = {
        "start": bt.get("bt_start"),
        "end": bt.get("bt_end"),
        "strategy_cum_return_pct": _pct100("cum_strategy"),
        "hs300_cum_return_pct": _pct100("cum_csi300"),
        "excess_annual_pct": _pct100("excess_annual"),
        "information_ratio": bt.get("info_ratio"),
        "excess_max_drawdown_pct": _pct100("excess_max_drawdown"),
        "strategy_max_drawdown_pct": _pct100("strategy_max_drawdown"),
        "slippage_bps": bt.get("slippage_bps"),
        "deal_price": bt.get("deal_price"),
    }
    metrics["backtest"].update({k: v for k, v in overlay.items() if v is not None})
    metrics["backtest"].setdefault("topk", config.TOPK)
    if meta.get("data_version"):
        metrics["data_version"] = meta["data_version"]


def _save_metrics() -> None:
    log_file = LOG_DIR / "train.log"
    if not log_file.exists():
        return
    metrics = parse_train_metrics(log_file.read_text(encoding="utf-8", errors="replace"))
    _merge_run_meta(metrics)
    metrics["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    METRICS_FILE.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_metrics() -> dict | None:
    """读取最近一次训练的指标；若 json 缺失但 train.log 还在则补解析一次。"""
    if not METRICS_FILE.exists() and (LOG_DIR / "train.log").exists():
        _save_metrics()
    if not METRICS_FILE.exists():
        return None
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 任务控制
# ---------------------------------------------------------------------------
def _worker(steps: list[str]) -> None:
    try:
        for step in steps:
            label, _ = STEPS[step]
            with _state.lock:
                _state.current_step = step
                _state.step_status[step] = "running"
            _append(f"===== 开始 [{step}] {label} =====")
            ok = _run_step(step)
            with _state.lock:
                _state.step_status[step] = "done" if ok else "failed"
                if not ok:
                    _state.error = f"{label} 执行失败，详见 output/pipeline_logs/{step}.log"
            _append(f"===== [{step}] {'完成' if ok else '失败'} =====")
            if not ok:
                return
        if "train" in steps:
            _save_metrics()
    finally:
        with _state.lock:
            _state.running = False
            _state.current_step = None
            _state.finished_at = time.time()


def start(steps: list[str] | None = None) -> list[str]:
    """启动管道任务；steps 为空则按序运行 fetch->dump->train。已在运行则抛错。"""
    steps = steps or DEFAULT_STEPS
    invalid = [s for s in steps if s not in STEPS]
    if invalid:
        raise ValueError(f"未知步骤: {invalid}，可选: {STEP_ORDER}")
    steps = [s for s in STEP_ORDER if s in steps]  # 固定执行顺序
    with _state.lock:
        if _state.running:
            raise RuntimeError("已有管道任务在运行中，请等待完成")
        _state.running = True
        _state.current_step = None
        _state.step_status = {s: "pending" for s in steps}
        _state.started_at = time.time()
        _state.finished_at = None
        _state.error = None
        _state.log.clear()
    threading.Thread(target=_worker, args=(steps,), daemon=True).start()
    return steps


def status() -> dict:
    with _state.lock:
        return {
            "running": _state.running,
            "current_step": _state.current_step,
            "steps": dict(_state.step_status),
            "started_at": _fmt_ts(_state.started_at),
            "finished_at": _fmt_ts(_state.finished_at),
            "error": _state.error,
            "log_tail": list(_state.log)[-300:],
        }


def get_step_log(step: str, tail: int = 2000) -> str | None:
    if step not in STEPS:
        return None
    f = LOG_DIR / f"{step}.log"
    if not f.exists():
        return ""
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:])
