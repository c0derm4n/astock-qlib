# 剔除纳指后的30只ETF研究

本分支将原池513100从交易行情、资格、特征和信号中剔除，固定使用其余30只ETF。原31只研究、行情归档、正式策略和账户分别保留。

研究计划与各轮失败原因：[SP500_EX_NASDAQ_PLAN.md](../../docs/SP500_EX_NASDAQ_PLAN.md)。完整验证、独立重放和图表检查已完成：[SP500_EX_NASDAQ_ACCEPTANCE.md](../../docs/SP500_EX_NASDAQ_ACCEPTANCE.md)。

主候选在第八轮运行前固定为60%通信ETF（515880）、35%黄金ETF（518880）、至少5%现金；通信不符合当日资格时，使用合格沪深300ETF（510300）承接，否则现金。历史研究通过不能替代实盘投资准入。

```powershell
# 校验冻结代码、数据、配置、验证产物后独立重放
.venv\Scripts\python.exe -m research.ex_nasdaq.replay

# 完整验证：两个起点、6个执行压力、5组权重和附加检查
.venv\Scripts\python.exe -m research.ex_nasdaq.validate --protocol research/ex_nasdaq/validation_protocol.json

# 根据完整通过的验证产物生成报告、曲线与冻结配置
.venv\Scripts\python.exe -m research.ex_nasdaq.report

# 复核冻结实验、基准算术、全部成交、规则拒绝和时期依赖；生成最终核验记录
.venv\Scripts\python.exe -m research.ex_nasdaq.verify_final

# 按登记文件重做候选探索；完整保留每个候选结果
.venv\Scripts\python.exe -m research.ex_nasdaq.explore --registry research/ex_nasdaq/candidates_round8.json
```

结果写入`output/sp500_ex_nasdaq`。8轮95个登记配置回放包括重复对照和失败结果；行业选择、参数选择均属于已查看历史后的开发研究。冻结文件变化会使旧入选策略重放被拒绝，需登记新的实验。

历史门槛通过，但结果高度依赖2025年通信行情：该年策略收益96.96%；截至2024年底长期年化仅7.99%，同期人民币SPY约16.00%。完整历史年化15.51%、回撤26.66%不能证明可持续超额收益，实盘投资准入仍未通过。

`models.py`另行训练季度滚动Ridge/LightGBM：只使用30只ETF过去3年已成熟的20日标签，固定超参数；模型方案是失败或部分区间达标的对照，最终固定配置并不调用模型预测。
