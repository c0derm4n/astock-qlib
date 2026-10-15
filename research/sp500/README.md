# 纳指65%＋黄金30%策略：冻结历史归档

**2026-10-13按用户要求归档并冻结，状态为 `archived_frozen`。** 投资标的仅限非美股，纳指65%＋黄金30%＋至少5%现金方案不再是当前候选、后续优化方向或待准入策略，不允许新增纳指配置探索、晋级生产或用于实盘。归档范围及保留文件SHA-256见 [archive_status.json](archive_status.json)。

原候选登记、冻结协议、`selected_strategy.json`、行情、基准和历史结果原样保留，供历史审计复核；共享计算工具仍可用于其他非美股研究。下面命令属于原实验入口记录，不能用于重写冻结归档或恢复活动研究。SPY仍可作为报告参考，不能作为交易资产。

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

两个主区间的历史收益接近目标及投资准入未通过的结论保留。该策略因非美股范围限制已归档，不再推进前瞻观察或投资准入。
