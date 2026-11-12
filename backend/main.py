"""FastAPI 入口：正式三策略服务与开发后台。

启动（在项目根目录）：
    uvicorn backend.main:app --host 127.0.0.1 --port 8000

生产模式下若 frontend/dist 存在，会自动托管前端静态文件（单端口访问）。
注意：任务状态保存在内存中，只能用单进程(单 worker)运行。
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import universe
from backend import BASE_DIR, data_service, pipeline
from src.utils import to_qlib_symbol
from backend.strategy_api import router as strategy_router
from backend.development_api import router as development_router

app = FastAPI(title="A股ETF三策略与开发后台 API", version="3.4.0")
app.include_router(strategy_router)
app.include_router(development_router)


@app.exception_handler(data_service.ArtifactUnavailableError)
async def artifact_unavailable(request: Request, exc: data_service.ArtifactUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "2"})

# 开发模式前端跑在 vite(5173)，放开跨域；生产由本服务托管静态文件，无跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    """内置 ETF 池（universe.py）；标注资产类别与是否可交易。

    可交易以当前运行的逐日资格审计为准；缺少结果时不放行。"""
    saved_tradable = data_service.read_run_meta().get("tradable")
    tradable = set(saved_tradable) if isinstance(saved_tradable, list) else None
    rows = []
    for code, name in universe.UNIVERSE.items():
        symbol = to_qlib_symbol(code)
        cls = universe.get_asset_class(code)
        if tradable is not None:
            ok = symbol in tradable
        else:
            ok = False
        rows.append({"code": code, "name": name, "symbol": symbol,
                     "asset_class": cls, "tradable": ok})
    return rows


@app.get("/api/config")
def get_config() -> dict:
    """关键全局配置（config.py），只读展示；口径与当前生产逻辑对齐。"""
    protocol = data_service.read_run_meta().get("protocol", {})
    return {
        "research_protocol": protocol,
        "configuration_source": "current_run" if protocol else "legacy_config",
        "market": config.MARKET,
        "start_date": config.START_DATE,
        "label_horizon": protocol.get("label_horizon", config.LABEL_HORIZON),
        # 切分口径：walk_forward=True 时逐年滚动重训，train/valid/test 不生效
        "walk_forward": config.WALK_FORWARD,
        "wf_train_start": config.WF_TRAIN_START,
        "wf_test_years": list(config.WF_TEST_YEARS),
        "train_period": list(config.TRAIN_PERIOD),
        "valid_period": list(config.VALID_PERIOD),
        "test_period": list(config.TEST_PERIOD),
        "topk": protocol.get("topk", config.TOPK),
        "n_drop": protocol.get("n_drop", config.N_DROP),
        "hold_thresh": protocol.get("hold_days", config.HOLD_THRESH),
        "open_cost": protocol.get("commission", config.OPEN_COST),
        "close_cost": protocol.get("commission", config.CLOSE_COST),
        "slippage_bps": protocol.get("slippage", config.SLIPPAGE_BPS) * 1e4,
        "deal_price": config.DEAL_PRICE,
        "exclude_qdii": config.EXCLUDE_QDII,
        "use_dynamic_universe": config.USE_DYNAMIC_UNIVERSE,
        "decision_run_time": config.DECISION_RUN_TIME,
    }


# ---------------------------------------------------------------------------
# 选 ETF 清单（对应 python -m src.predict）
# ---------------------------------------------------------------------------
@app.get("/api/picks")
def get_picks(
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    topk: int = Query(config.TOPK, ge=1, le=50),
) -> dict:
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


@app.get("/api/acceptance")
def acceptance() -> dict:
    report = data_service.read_acceptance()
    if report is None:
        raise HTTPException(404, "尚无审计研究报告，请先运行训练")
    return report


# ---------------------------------------------------------------------------
# 数据管道（对应 README 快速开始的三步命令）
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    steps: list[str] | None = None  # None/空 = 按序运行 fetch->train


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
def pipeline_logs(step: str, tail: int = Query(2000, ge=1, le=10000)) -> str:
    text = pipeline.get_step_log(step, tail)
    if text is None:
        raise HTTPException(400, f"未知步骤 {step}，可选: {pipeline.STEP_ORDER}")
    return text


# ---------------------------------------------------------------------------
# 生产模式：托管前端构建产物（frontend/dist 存在时；哈希路由无需重写规则）
# 必须放在所有 /api 路由之后挂载
# ---------------------------------------------------------------------------
_DIST = BASE_DIR / "frontend" / "dist"
# 旧接口保留兼容历史脚本，新的开发后台只暴露正式双策略接口文档。
for _route in app.routes:
    _path = getattr(_route, 'path', '')
    if _path.startswith('/api/') and _path != '/api/health' and not _path.startswith(('/api/strategies', '/api/development')):
        _route.include_in_schema = False
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")
