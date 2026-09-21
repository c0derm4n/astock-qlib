# v2 前后端部署与运维

本项目由 Vue 3 界面、FastAPI 本地服务、DuckDB 行情库和版本化研究产物组成。Web 的训练按钮运行完整研究与验收；模拟决策按钮执行 `src.decide --paper`，只生成订单建议。

## 安装与启动

在项目根目录使用 Python 3.10+ 的虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend/requirements-dev.txt
cd frontend
npm ci
npm run build
cd ..
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`；接口文档为 `/docs`。后端自动托管 `frontend/dist`，前端使用哈希路由。开发时可在 `frontend` 运行 `npm run dev`，5173端口将 `/api` 代理到8000端口。

使用单 worker；管道状态在内存中，同一时间仅允许运行一个任务，重复触发返回409。当前服务用于本地操作，尚无多用户权限管理，跨主机部署需要另行配置认证与访问控制。

## 数据与研究更新

1. 运行 `python -m src.fetch_data`。默认auto模式：本机有令牌使用原授权服务，没有令牌使用腾讯原价与新浪公司行动的免费增量。免费源逐只核对历史重叠、因子衔接和覆盖；通过后备份并原子切换数据库，不支持--force建立全历史。
2. 需要Qlib兼容数据时运行 `python -m src.dump_qlib`，按历史逐日资格导出，并保留旧目录备份。
3. 运行 `python -m src.train`，完成滚动训练、对照、压力、参数邻域和独立生产训练。仅成功完成后切换 `output/current_run.json`。
4. 在 `/api/acceptance` 或总览查看投资准入结果。训练进程成功只表示研究运行完成，不表示投资准入通过。

每次研究写入 `output/runs/<run_id>/`，包含模型、预处理器、数据快照、代码副本、协议、哈希、报告及验收结果。CLI预测和Web从当前指针读取。Qlib导出单独写 `output/data_meta.json`；管道日志在 `output/pipeline_logs/`。

代码或协议变更后须重训验收，否则决策拒绝加载旧版本。仅重启后端无法替代重新验收。`src.research --quick` 为开发冒烟，不能通过投资门禁。

## 账户、日常决策与观察

新模拟账户可运行 `python -m src.account init --cash 1000000`；已有账户必须核对份额与现金，初始化不能覆盖原文件。完整格式及成交导入示例见根目录README。

交易日正式收盘数据完成后（默认19:00），运行 `python -m src.decide --paper`，生成下一交易日开盘模拟订单。使用交易所日历处理周末、节假日；不抓取14:30快照、不自动更新持仓。定时脚本为 `run_decide_1900.ps1`，需由使用者按需要配置任务计划，项目不会自动注册。

成交回报通过 `src.account apply-fills` 导入；分红、折算、清盘回款通过 `apply-actions` 根据核对回单入账。执行日19:00后运行 `python -m src.forward`，记录当天前瞻观察。规则版本、信号冻结、连续交易日、账户净值衔接及成交完成率都参与检查；历史回填不计入前瞻。

## 故障与验收

- 缺少预测/报告返回404；损坏或更新中的产物返回503，附带重试提示。
- 账户缺失、损坏或旧代码列表格式时停止决策；不会默认为空仓。
- 行情或模型过期、协议/代码哈希变化、准入未通过时，正式建议会被阻断。`--paper` 用于检查研究结果，不能证明实盘可用。
- 子进程失败后管道标记失败并跳过后续步骤；后端重启不恢复原任务历史。
- 修改数据源或标的目录后，应补齐原始价、复权因子、公告日期及生命周期证据。缺少完整清盘/退市目录或公司行动证据时，不打开审计门禁。

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q src backend tests
cd frontend
npm run build
```

最新结果见 [v2.1.1投资准入验收报告](INVESTMENT_READINESS_ACCEPTANCE.md)。免费源更新明细在output/data_updates；缓存和前端体积性能对照保存在 [工程优化记录](OPTIMIZATION_PLAN.md)。
