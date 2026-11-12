"""开发后台只读接口，全部基于正式三策略的可绑定运行快照。"""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from backend import development_service

router = APIRouter(prefix='/api/development', tags=['开发后台 · 正式周频'])
StrategyId = Literal['csi300_pe_equal', 'trend200_vol15', 'dividend_band_a']
RunId = Query(default=None, pattern=r'^\d{8}-\d{6}-[a-f0-9]{8}$')


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except development_service.DevelopmentUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, '正式策略不存在') from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get('/overview')
def overview(run_id: str | None = RunId):
    return _call(development_service.overview, run_id)


@router.get('/universe')
def universe(strategy_id: StrategyId, run_id: str | None = RunId):
    return _call(development_service.get_universe, strategy_id, run_id)


@router.get('/picks/dates')
def dates(strategy_id: StrategyId, run_id: str | None = RunId):
    return _call(development_service.available_dates, strategy_id, run_id)


@router.get('/picks')
def picks(strategy_id: StrategyId, date: str | None = Query(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$'),
          topk: int = Query(default=29, ge=1, le=29), run_id: str | None = RunId):
    return _call(development_service.get_picks, strategy_id, date, topk, run_id)


@router.get('/backtest')
def backtest(strategy_id: StrategyId, run_id: str | None = RunId):
    return _call(development_service.backtest, strategy_id, run_id)
