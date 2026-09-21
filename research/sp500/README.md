# 固定31只ETF的标普500对标研究

最终方案、风险及全部失败实验见 [中文验收报告](../../docs/SP500_STRATEGY_ACCEPTANCE.md)。这是独立开发研究，输出保存到 `output/sp500_exploration`；正式训练、发布指针和账户没有自动切换。

```powershell
# 重放选定规则，核对数据、代码和冻结配置
.venv\Scripts\python.exe -m research.sp500.replay

# 重做冻结的完整压力、权重邻域与两个主区间验证
.venv\Scripts\python.exe -m research.sp500.validate --protocol research/sp500/validation_protocol_v3.json

# 根据最新验证产物重新生成报告和曲线
.venv\Scripts\python.exe -m research.sp500.report
```

`selected_strategy.json` 固定65%纳指ETF、30%黄金、5%现金；纳指不合格时使用合格恒生ETF代理。观察波动但不以波动择时，仓位、集中度、流动性、当时已公告规模、费用和次日成交限制仍生效。具体实现为 `explore.py` 的分配回调及共享的 `src.simulation` / `src.portfolio` 执行规则。

6轮候选登记在 `candidates*.json`，3轮完整验证登记在 `validation_protocol*.json`。34个含基线候选及不利结果保留；最新完整验证为 `20260913-090253`。重放依赖本机保留的行情、NAV、元数据和基准快照；修改冻结输入或规则会被拒绝，需另外登记研究。

```powershell
# 可选：取得新的独立基准快照，不改变ETF正式行情库
.venv\Scripts\python.exe -m research.sp500.benchmark --end 2026-09-11

# 重新运行指定登记批次；不会发布正式模型
.venv\Scripts\python.exe -m research.sp500.explore --registry research/sp500/candidates_round6_long.json
```

两个主区间的收益接近目标已验证；完整历史审计、126个真实前瞻日及原20%回撤准入要求仍未满足，不能将研究标签当作实盘批准。
