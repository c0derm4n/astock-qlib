"""FastAPI 入口：把 README 的四步管道与结果查询包装成 HTTP 接口。

启动（在项目根目录）：
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

生产模式下若 frontend/dist 存在，会自动托管前端静态文件（单端口访问）。
注意：任务状态保存在内存中，只能用单进程(单 worker)运行。
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import universe
from backend import BASE_DIR, data_service, pipeline
from src.utils import to_qlib_symbol

app = FastAPI(title="A股ETF轮动系统 API", version="1.0.0")

# 开发模式前端跑在 vite(5173)，放开跨域；生产由本服务托管静态文件，无跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 基础信息
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/overview")
def overview() -> dict:
    return data_service.overview()


@app.get("/api/universe")
def get_universe() -> list[dict]:
    """内置 ETF 池（universe.py）。"""
    return [
        {"code": code, "name": name, "symbol": to_qlib_symbol(code)}
        for code, name in universe.UNIVERSE.items()
    ]


@app.get("/api/config")
def get_config() -> dict:
    """关键全局配置（config.py），只读展示。"""
    return {
        "market": config.MARKET,
        "start_date": config.START_DATE,
        "label_horizon": config.LABEL_HORIZON,
        "train_period": list(config.TRAIN_PERIOD),
        "valid_period": list(config.VALID_PERIOD),
        "test_period": list(config.TEST_PERIOD),
        "topk": config.TOPK,
        "n_drop": config.N_DROP,
        "open_cost": config.OPEN_COST,
        "close_cost": config.CLOSE_COST,
    }


# ---------------------------------------------------------------------------
# 选 ETF 清单（对应 python -m src.predict）
# ---------------------------------------------------------------------------
@app.get("/api/picks")
def get_picks(date: str | None = None, topk: int = config.TOPK) -> dict:
    """某交易日模型打分 TopK 清单；date 缺省=最新，非交易日回退到之前最近交易日。"""
    if not data_service.has_predictions():
        raise HTTPException(404, "暂无预测结果，请先在「数据管道」运行训练(train)")
    try:
        return data_service.get_picks(date, topk)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/picks/dates")
def get_pick_dates() -> list[str]:
    """有预测数据的交易日列表（用于前端日期选择器）。"""
    if not data_service.has_predictions():
        return []
    return data_service.available_dates()


# ---------------------------------------------------------------------------
# 回测（对应 train 写出的 backtest_report.csv）
# ---------------------------------------------------------------------------
@app.get("/api/backtest/report")
def backtest_report() -> dict:
    if not data_service.has_backtest():
        raise HTTPException(404, "暂无回测报告，请先运行训练(train)")
    return data_service.backtest_series()


@app.get("/api/backtest/summary")
def backtest_summary() -> dict:
    if not data_service.has_backtest():
        raise HTTPException(404, "暂无回测报告，请先运行训练(train)")
    return data_service.backtest_summary()


# ---------------------------------------------------------------------------
# 每日决策（对应 python -m src.decide 的输出 output/decision_YYYYMMDD.csv）
# ---------------------------------------------------------------------------
@app.get("/api/decision/latest")
def decision_latest() -> dict:
    d = data_service.latest_decision()
    if d is None:
        raise HTTPException(404, "暂无决策清单，请先在「数据管道」运行 每日决策(decide)")
    return d


# ---------------------------------------------------------------------------
# 训练指标（最近一次 train 的 IC / 回测打印解析）
# ---------------------------------------------------------------------------
@app.get("/api/train/metrics")
def train_metrics() -> dict:
    metrics = pipeline.read_metrics()
    if metrics is None:
        raise HTTPException(404, "暂无训练指标，请先运行训练(train)")
    return metrics


# ---------------------------------------------------------------------------
# 数据管道（对应 README 快速开始的四步命令）
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    steps: list[str] | None = None  # None/空 = 按序运行 fetch->dump->train


@app.post("/api/pipeline/run")
def pipeline_run(req: RunRequest) -> dict:
    try:
        steps = pipeline.start(req.steps)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"started": steps}


@app.get("/api/pipeline/status")
def pipeline_status() -> dict:
    return pipeline.status()


@app.get("/api/pipeline/logs/{step}", response_class=PlainTextResponse)
def pipeline_logs(step: str, tail: int = 2000) -> str:
    text = pipeline.get_step_log(step, tail)
    if text is None:
        raise HTTPException(400, f"未知步骤 {step}，可选: {pipeline.STEP_ORDER}")
    return text


# ---------------------------------------------------------------------------
# 生产模式：托管前端构建产物（frontend/dist 存在时；哈希路由无需重写规则）
# 必须放在所有 /api 路由之后挂载
# ---------------------------------------------------------------------------
_DIST = BASE_DIR / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")
