"""仅本地、无券商委托的正式双策略 API。"""
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from backend import strategy_service

router = APIRouter(prefix='/api/strategies', tags=['正式双策略'])


def local_mutation(request):
    origin = request.headers.get('origin')
    if request.headers.get('x-local-request') != '1':
        raise HTTPException(403, '本地操作需携带 X-Local-Request: 1')
    if origin:
        parsed = urlparse(origin)
        if parsed.hostname not in ('localhost', '127.0.0.1', '::1') or parsed.scheme not in ('http', 'https'):
            raise HTTPException(403, '仅接受本地页面发起的操作')


class RefreshRequest(BaseModel):
    fetch: bool = True


class Holding(BaseModel):
    symbol: str = Field(pattern=r'^(SH|SZ)\d{6}$')
    quantity: int = Field(ge=0, le=10000000000)
    holding_days: int = Field(ge=0, le=10000, default=0)


class AccountRequest(BaseModel):
    run_id: str = Field(pattern=r'^\d{8}-\d{6}-[a-f0-9]{8}$')
    cash: float = Field(ge=0, le=1e12, allow_inf_nan=False)
    holdings: list[Holding] = Field(default_factory=list, max_length=29)

    @model_validator(mode='after')
    def unique(self):
        if len({h.symbol for h in self.holdings}) != len(self.holdings):
            raise ValueError('持仓代码不能重复')
        return self


@router.get('')
def all_strategies():
    return strategy_service.snapshot()


@router.get('/refresh/status')
def refresh_status():
    return strategy_service.status()


@router.post('/refresh')
def refresh(req: RefreshRequest, request: Request):
    local_mutation(request)
    try:
        return strategy_service.start(req.fetch)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@router.post('/{id_}/preview')
def account(id_: str, req: AccountRequest, request: Request):
    local_mutation(request)
    try:
        return strategy_service.account_preview(id_, req.run_id, req.cash, [h.model_dump() for h in req.holdings])
    except KeyError:
        raise HTTPException(404, '策略不存在')
    except ValueError as exc:
        raise HTTPException(400, str(exc))
