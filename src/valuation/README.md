# 等权估值正式计算模块

2026-11-09根据用户明确授权，从已通过独立账本核验的`real-etfrun-equal-v2`迁入。`promotion.json`记录迁入时五个模块的SHA256；初始迁入逐字节一致，研究源码与协议冻结保留。正式代码不导入研究目录。

服务适配与增量数据在`src/formal_valuation.py`，正式参数在根目录`valuation_strategy_protocol.json`，真实历史种子位于`data/formal/valuation/seed/`。部署本地工作区时应同时保留种子manifest及其列出的CSV；只复制源码不能获得真实历史数据。

每次正式构建都冻结独立的`valuation_data`输入、回放`valuation_cases`账本，并从原价、成交与公司行动独立复核后才发布四策略指针。来源只追加新日值；供应商修改旧日值或缺少日频覆盖时保留旧值、记录错误，不静默重写历史。ETF原价取正式库增量；每手交易量转换为份额，新增公司行动须具备登记及支付日。

`tests/test_formal_valuation.py`核对每日净值与冻结研究相等、独立账本、信号订单数量、时点延迟、历史修订拒绝、刷新失败降级和旧双策略启动升级；原策略回归另见`tests/test_formal_service.py`。等权PB未证实为MRQ，数据为历史重建非PIT，服务登记不代表达到投资准入。
