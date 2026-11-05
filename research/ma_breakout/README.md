# MA5/MA10 4% 通信黄金研究

本目录是独立离线研究线，不修改 `src/`、正式策略、生产指针、正式数据库或旧冻结实验。

## 当前状态

- 最终协议：`ma-breakout-20261023-3`
- 最终不可覆盖 run：`output/ma_breakout/runs/20261023-113723-47fb0e5d`
- 区间：2016-09-12—2026-10-19，中国交易日 2448 行，初始资金 100 万元
- 执行口径：收盘确认、下一交易日开盘；T+2 只作为延迟压力
- 历史结论：主方案 CAGR 2.34%、最大每日净值回撤 16.54%、最差完整年 2018 年 -4.61%
- 联合目标未通过：年化低于周频和日频 MA200；对周频，最差完整年更小但全期最大每日净值回撤略高，两项风险指标不能擅自排序；对日频，CAGR、最差完整年和最大每日净值回撤三项均较差。2026 为不足年，收益 -5.07%。不是生产候选或实盘准入结果

v3 是在已查看 v1/v2 完整 16 场景后，对清仓后入场语义的修正复核，不是首次查看结果前的预注册，也不是独立样本外验证。参数未根据 v1/v2 结果调整。

## 文件

- `engine.py`：基于 `src/formal_cash.simulate_cash` 的独立现金引擎扩展。显式信号日、T+1/T+2、在途委托冻结、公司行动撤单、订单结果日志。
- `signals.py`：各 ETF 自身复权价格的 MA5/MA10/1日或2日4%因果状态机、通信资格替代、止损意图、正式风险/漂移/换手约束。
- `protocol.json`：v3 最终协议、场景、指标、登记时序和限制。
- `run.py`：协议登记、输入哈希冻结、16 场景运行、独立账本审计、图表与中文报告 CLI。
- `test_engine.py`、`test_signals.py`、`test_run.py`：现金、信号、执行、公司行动和统计回归测试。

## 命令

```powershell
# 23项合成测试
.venv\Scripts\python.exe -m unittest `
  research.ma_breakout.test_engine `
  research.ma_breakout.test_signals `
  research.ma_breakout.test_run -v

# 新登记不可覆盖 run
.venv\Scripts\python.exe -m research.ma_breakout.run --register

# 完整运行已登记目录
.venv\Scripts\python.exe -m research.ma_breakout.run --run <run目录>

# 只跑主方案、无波动辅助与两条基线的基础情景
.venv\Scripts\python.exe -m research.ma_breakout.run --run <run目录> --quick
```

每个 case 输出 report、trades、actions、holdings、orders、signals、drawdown CSV 和 case JSON。run 根目录另含 `results.json`、`completion.json`、`report.md`、`net_value.png` 与 `drawdown.png`。

## 规则摘要

- 入场：`(r1 >= threshold or r2 >= threshold) and P >= MA5`
- 退出：`P < MA10`，同日退出优先
- 持仓在没有退出时延续；跌破 MA5 不单独卖出
- 通信、黄金分别 60%、35%，至少 5% 现金
- 通信市场资格失效时才允许沪深300替代；替代资产只在自己的有效资格窗口累积信号
- 主方案直接复用 `formal_rules.bounded_targets` 的波动、权重和30%共享换手约束
- 无波动辅助只关闭波动缩放，其他约束不变
- T+2期间一个标的最多一组在途单；普通非零目标变化不重发，明确零目标才撤未成交买单
- 清仓卖单不因新入场信号撤销；公司行动撤单
- 实际清零后的同一收盘，只要当日入场条件和资格满足且无退出，即可重新入场

## 勘误

- `output/ma_breakout/errata/20261023-opening-event-count.json`：v1 开仓事件统计漏计拆分后的持仓状态。
- `output/ma_breakout/errata/20261023-post-clear-entry.json`：v2 在清仓后额外要求 `entry_event`，错误影响 T+2 压力路径。

v1、v2 及其全部哈希保留，不覆盖。报告与结论以 v3 为准。
