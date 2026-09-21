# A股ETF轮动研究系统 v2.1.1 — Web界面

Vue 3 + Vite + Element Plus + ECharts，数据经 FastAPI `/api` 查询。前端不直接读取数据文件。

| 页面 | 路由 | 内容 |
| --- | --- | --- |
| 总览 | `/#/` | 投资准入阻断、真实前瞻天数、预测质量、复利指标与参考策略 |
| 选ETF清单 | `/#/picks` | 当前运行的历史打分；非交易日回退到之前最近有预测的日期 |
| 每日决策 | `/#/decision` | 收盘信号对应的次日开盘模拟订单份额、参考价、目标权重和账户状态 |
| 回测分析 | `/#/backtest` | 扣费净值、CAGR、复利回撤、252日年化IR、沪深300ETF参考 |
| 数据管道 | `/#/pipeline` | fetch → dump → train；19:00后模拟决策单独触发，失败与跳过状态、日志 |

沪深300ETF参考采用95%仓位上限并计入成本和建仓过程；等权和股债金参考同样受各自明确的组合规则约束。模型打分与回测通过不等于投资准入通过。

```powershell
npm ci
npm run dev    # 5173，需同时运行8000端口后端
npm run build  # frontend/dist由后端托管
```

部署方式见 [部署文档](../docs/DEPLOYMENT.md)，最新验证结果见 [v2.1.1投资准入验收报告](../docs/INVESTMENT_READINESS_ACCEPTANCE.md)；[v2.0量化验收报告](../docs/QUANT_ACCEPTANCE.md) 保留为历史记录。

界面使用哈希路由与按页懒加载；ECharts只引入折线图及必要组件；轮询避免重叠请求，离开页面时取消请求并释放图表。第三方包仍有体积警告，构建成功与页面验收分别记录。

`docs/images`中的旧截图属于v1演示样例，不代表当前模型收益或v2验收状态。
