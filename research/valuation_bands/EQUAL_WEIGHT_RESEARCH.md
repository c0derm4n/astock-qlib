# 等权估值版真实历史回测说明

登记日期：2026-11-09。协议：`valuation-bands-etfrun-equal-20261109-v1`。

用户明确授权另建等权版本并标明差异。原`protocol.json`整体法PE_TTM/PB_MRQ及其数据阻塞结论保持不变。本版本为独立研究，不切换生产。

## 数据与规则

采用ETF.run公开展示的[沪深300日频历史](https://www.etf.run/index/SSE/000300)与[创业板指日频历史](https://www.etf.run/index/SZSE/399006)，归档HTML后只提取日期、收盘、`equalWeightedPeTtm`与`equalWeightedPbTtm`。两个指数2011-01-01—2026-11-06共3847个交易日价格和所需估值完整、有限且大于0。网站已经计算的全历史分位不进入交易信号。

原始HTML、提取结果及下载核查位于`data/valuation_bands/raw/continuation_20261109/`；规范化快照位于`data/valuation_bands/snapshots/etfrun-equal-20261109-v1/`，SHA256及来源写在manifest。

这是网站声明的等权口径，不是整体法。PE字段存为`PE_TTM_EQUAL`；PB字段存为`PB_ETFRUN_EQUAL`，保留网站字段名和“MRQ未经证实”的财务口径备注，不将它改名为`PB_MRQ`。供应商未公开完整财报、历史成分和修订算法，因此证据等级为`historical_reconstructed`；下一交易日19:00可得的延迟假设不消除历史重述偏差。

除估值数据/口径外，沿用原方案最近五个完整自然年低点校准、三批建仓、20交易日冷却、85%/75%退出分位、60%/40%权益上限、周首19:00评估和次日开盘执行等约定。ETF价格和公司行动复用前次审查的专属研究快照；部分公告/登记/支付时间仍有原研究假设。

基准对齐已有正式策略，冻结账户区间为2016-09-12—2026-11-04，共2460个交易日，各初始100万元。2016和2026为不足年，不参与最差完整年。仓位上限是决策约束，价格漂移及周频执行可能使实际持仓短期超限，不将它当作实时硬上限。

## 预先登记与产物

首次回放前登记两个主策略和四个执行压力（滑点翻倍、T+2各两条），不进行绩效驱动调参。原25场景的敏感性和缺数据辅助模块不混入这六个case。股债比较、红利低波配置仍缺合格数据，本版只回放两条现金主策略。

- 参数协议：`protocol_etfrun_equal_weight.json`。
- 数据解析：`etfrun_data.py`；新增解析、重复主键和独立协议哈希测试。
- 最终冻结run：`output/valuation_bands/runs/real-etfrun-equal-v2/`。
- 四策略比较和报告：`output/valuation_bands/equal_weight_20261109/`。
- 首次run `real-etfrun-equal-v1`保留。该批发现买入持有基准对2021-02-08创业板ETF缺价按0估值；修复为最多5交易日沿用原价并标记，超过期限停止。v2重跑全部六个case及基准，两条主策略的每日净值与v1逐字节相同。

复跑数据与登记（运行目录须使用新名字）：

```powershell
.venv/Scripts/python.exe -m research.valuation_bands.run check-data --protocol research/valuation_bands/protocol_etfrun_equal_weight.json --manifest data/valuation_bands/snapshots/etfrun-equal-20261109-v1/manifest.json
.venv/Scripts/python.exe -m research.valuation_bands.run register --protocol research/valuation_bands/protocol_etfrun_equal_weight.json --manifest data/valuation_bands/snapshots/etfrun-equal-20261109-v1/manifest.json
.venv/Scripts/python.exe -m research.valuation_bands.run run --directory <新登记目录> --group all
.venv/Scripts/python.exe -m research.valuation_bands.verify --directory <新登记目录>
.venv/Scripts/python.exe -m research.valuation_bands.audit --research-run <新登记目录> --output <新比较目录>
```

这次两策略都只有两个已经卖清的周期，不足以宣称统计上稳健或未来风险可控。最终数值以落盘账本独立复核和比较报告为准，不引用截图自述或合成收益。
