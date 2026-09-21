# research/ 独立研究线

本目录是**离线研究区**，不在生产运行路径上。三条研究线各自独立登记、独立重放、独立产出，从不写 `output/current_run.json`，也不切换正式模型、每日入口或账户。

## 与主系统的关系

| 方向 | 状态 | 说明 |
| --- | --- | --- |
| `research/` → `src/`、`backend/` | 允许 | 复用 `backend.artifacts.write_json_atomic`、`src.portfolio`、`src.simulation`、`src.market`、`src.universe_filter`、`src.models`、`src.account` |
| `src/`、`backend/` → `research/` | 禁止 | 已核实零引用。改造或删除本目录不影响生产链路 |

生产入口（`python -m src.train`、`python -m src.decide`、`/api/*`）都不经过本目录。

## 目录

| 文件夹 | 定位 | 输出目录 |
| --- | --- | --- |
| [sp500/](sp500/README.md) | 固定31只ETF对标标普500的探索与验证 | `output/sp500_exploration/` |
| [ex_nasdaq/](ex_nasdaq/README.md) | 剔除纳指后30只ETF的独立历史研究 | `output/sp500_ex_nasdaq/` |
| [goal30/](goal30/README.md) | 30只ETF目标的核验、每日观察与周度双档目标 | `output/goal30/` |

三条线的演进关系：`sp500` 先确立「海外权益 + 黄金」的对标思路 → `ex_nasdaq` 剔除纳指、在30只池上重做 → `goal30` 在30只池上做逐日核验、现金分红回放与公司行动取证，最后由 `goal30/weekly_goal.py` 汇总为周频双档目标研究。

## 共同方法

三条线使用同一套研究纪律，具体文件名各有取舍：

1. **事前登记**——候选批次与验收标准先写进 `*_protocol.json` / `candidates*.json`，再运行。
2. **冻结与重放**——入选方案写入 `selected_strategy.json`，含源文件与数据哈希。改动任何冻结输入都会让旧重放被拒绝，必须另外登记新实验。这不是缺陷，是设计。
3. **失败候选保留**——登记批次中失败的配置一并保留，不事后剔除。
4. **独立核查**——`verify_*.py` / `validate_*.py` 单独复核算术、逐笔成交、规则拒绝、时期依赖与原件指纹。
5. **输出隔离**——各写各的目录，互不覆盖。

## sp500/ — 固定31只ETF对标标普500

**定位**：独立开发研究。`selected_strategy.json` 固定 65% 纳指ETF、30% 黄金、5% 现金；纳指不合格时用合格恒生ETF代理。观察波动但不以波动择时，仓位、集中度、流动性、当时已公告规模、费用与次日成交限制仍然生效。

**内容**：6 轮候选登记（`candidates*.json`）、3 轮完整验证（`validation_protocol*.json`），34 个含基线与不利结果的候选全部保留。`benchmark.py` 只取独立基准快照，不改变 ETF 正式行情库。

```powershell
python -m research.sp500.replay
python -m research.sp500.validate --protocol research/sp500/validation_protocol_v3.json
python -m research.sp500.report
python -m research.sp500.benchmark --end 2026-09-11          # 可选：取新的独立基准快照
python -m research.sp500.explore --registry research/sp500/candidates_round6_long.json
```

**状态**：两个主区间的收益接近目标已验证；完整历史审计、126 个真实前瞻日及原 20% 回撤准入要求**未满足**。[验收报告](../docs/SP500_STRATEGY_ACCEPTANCE.md)

## ex_nasdaq/ — 剔除纳指后的30只ETF

**定位**：把 513100 从行情、资格、特征和信号中整体剔除，固定使用其余 30 只。原 31 只研究、行情归档、正式策略与账户分别保留。

**方案**：60% 通信ETF（515880）+ 35% 黄金ETF（518880）+ 至少 5% 现金；通信不符合当日资格时用合格沪深300ETF（510300）承接，否则现金。

**内容**：8 轮 95 个登记配置（`candidates_round1..8.json`），含重复对照与失败结果。`core.py` 提供 30 只池的市场与资格校验（被 `goal30` 复用）；`models.py` 另行训练季度滚动 Ridge/LightGBM，但**最终固定配置不调用模型预测**。

```powershell
python -m research.ex_nasdaq.replay
python -m research.ex_nasdaq.validate --protocol research/ex_nasdaq/validation_protocol.json
python -m research.ex_nasdaq.report
python -m research.ex_nasdaq.verify_final
python -m research.ex_nasdaq.explore --registry research/ex_nasdaq/candidates_round8.json
```

**状态**：历史门槛通过，但结果高度依赖 2025 年通信行情（该年策略收益 96.96%；截至 2024 年底长期年化仅 7.99%，同期人民币 SPY 约 16.00%）。完整历史年化 15.51%、回撤 26.66%，**不能证明可持续超额收益，实盘准入未通过**。[验收报告](../docs/SP500_EX_NASDAQ_ACCEPTANCE.md)

## goal30/ — 30只ETF目标的核验、每日观察与周度双档

**定位**：三条线中文件最多（72 个 Python 文件）的一条，也是唯一包含**每日实盘前观察**与**公司行动/上市目录取证**的一条。输出到 `output/goal30/`。

按用途分五类：

- **研究与回放**——`cash_replay.py`（登记权益、除息应收、支付到账、真实拆分的现金分红回放）、`risk_overlay.py`、`turnover_replay.py`、`relative_allocation.py`、`group_diversification.py`、`redeployment.py`、`remaining_dividend_*.py`、`action_sensitivity.py`、`primary_*.py`、`daily*.py`、`broad_*.py`
- **每日观察**——`daily.py`（北京时间 19:00 运行；休市或尚未到时不更新行情，成功观察同一日只冻结一次）、`forward.py`（前瞻观察的冻结与时效核算）
- **周度双档目标**——`weekly_goal.py` + `weekly_goal_protocol.json`。**本目录中直接对应「年化/回撤双档 + 周频调仓」目标的模块，也是三条线的汇总点**：它复用 `ex_nasdaq` 的30只池与冻结方案、`goal30` 的现金回放与风险叠加、`sp500` 的分配回调与相对指标。协议登记 6 个变体（趋势窗口 200/240 × 波动预算 10%–25%）× 3 个情景（基础、双倍滑点、支付延迟 5 日）共 18 个完整结果
- **独立核查**——11 个 `verify_*.py`（现金账本、分红公告原件、上市日期、执行对齐、风险叠加等）
- **本地测试与报告**——13 个 `test_*.py`、5 个 `report_*.py`

```powershell
python -m research.goal30.audit --restart        # 原规则哈希验证 + 95配置筛查 + 十年现金重建与账本审计
python -m research.goal30.daily --refresh        # 每日 19:00
python -m research.goal30.daily --account 用户核对的账户.json --refresh
python -m research.goal30.verify_cash <归档目录> --output <输出>
python -m research.goal30.weekly_goal --register              # 登记周频双档协议
python -m research.goal30.weekly_goal --run <输出目录>          # 运行已登记的周频双档研究
```

**状态**：现金分红回放（2016-09-12—2026-09-11）年化 15.97%、人民币 SPY 15.34%；年度最大回撤均值 13.23% vs 13.55%，日均回撤 7.48% vs 3.42%——**两种平均回撤口径给出不同结论，验收定义仍待确认**。仍缺历史目录、公司行动来源完整审计及 126 日真实前瞻，**合格前瞻天数为 0**。[现金回放验收](../docs/SP500_CASH_REPLAY_ACCEPTANCE.md)、[发行人正文补充](../docs/SP500_ISSUER_EVIDENCE_ACCEPTANCE.md)、[上市日期验收](../docs/SP500_LISTING_DATES_ACCEPTANCE.md)

## 测试覆盖

主测试套件中有 7 个文件覆盖这三条线：`tests/test_sp500_research.py`、`tests/test_ex_nasdaq.py`、`tests/test_goal30.py`、`tests/test_goal30_actions.py`、`tests/test_goal30_cash.py`、`tests/test_goal30_forward.py`、`tests/test_goal30_weekly.py`。另有 `goal30/` 内部的 13 个本地测试。

```powershell
python -m unittest discover -s tests -v
```

## 注意

- 三条线的**实盘投资准入均未通过**。历史门槛通过不等于可持续超额收益，研究标签不能当作实盘批准。
- 重放依赖本机保留的行情、NAV、元数据与基准快照；冻结文件（`selected_strategy.json`、`*_protocol.json`、快照）一旦改变，旧重放会被拒绝。
- `goal30` 的每日命令没有历史日期参数：过去日期的测试只能写入独立验证目录，不计作前瞻。
- 未提供账户时 `rebalance_needed=null`——不知道用户仓位，不能认定已空仓或需要买入。
