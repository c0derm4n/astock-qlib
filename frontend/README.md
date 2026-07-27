# A股ETF轮动系统 — Web 前端

基于 **Vue 3 + Vite + Element Plus + ECharts** 的单页应用，是根目录 CLI 管道（`src/` 四步）的可视化界面。数据全部来自 FastAPI 后端（`backend/`，`/api` 前缀），前端不直接读写任何数据文件。

## 页面

| 页面 | 路由 | 说明 |
|------|------|------|
| 总览 | `/#/` | IC/RankIC、回测关键指标、最新 TopK 清单 |
| 选ETF清单 | `/#/picks` | 按交易日/TopK 查询模型打分清单（等价 `python -m src.predict`，非交易日自动回退） |
| 每日决策 | `/#/decision` | 最新 `output/decision_YYYYMMDD.csv` 买入/卖出/持有清单 |
| 回测分析 | `/#/backtest` | 策略(扣费) vs 沪深300 累计收益曲线 + 汇总指标 |
| 数据管道 | `/#/pipeline` | 一键运行 fetch → dump → train；decide(每日14:30决策) 单独触发；实时滚动日志 |

技术要点：哈希路由（`createWebHashHistory`，便于 FastAPI 直接托管静态文件）；axios 拦截器统一弹错误提示；顶栏每 5 秒轮询管道状态。

## 目录结构

```
frontend/
├── index.html
├── package.json
├── vite.config.js     # dev 端口 5173，/api 代理到 127.0.0.1:8000
└── src/
    ├── main.js        # 入口：ElementPlus(中文) + router
    ├── App.vue        # 布局：侧边菜单 + 顶栏任务状态
    ├── api.js         # axios 实例（baseURL=/api，统一错误提示）
    ├── router.js
    └── views/
        ├── DashboardView.vue  # 总览
        ├── PicksView.vue      # 选ETF清单
        ├── DecisionView.vue   # 每日决策（买/卖/持有清单）
        ├── BacktestView.vue   # 回测分析（ECharts）
        └── PipelineView.vue   # 数据管道（任务控制 + 日志）
```

## 开发

```powershell
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173 ，需同时启动后端(8000)
```

后端启动方式见根目录 `docs/DEPLOYMENT.md`；接口文档 `http://127.0.0.1:8000/docs`。

## 构建（生产）

```powershell
npm run build      # 产出 frontend/dist/
```

`backend/main.py` 检测到 `frontend/dist` 存在即自动托管，之后只启动后端、浏览器访问 `http://127.0.0.1:8000` 即可（单端口，无跨域）。Nginx 托管与常驻部署见 `docs/DEPLOYMENT.md`。

## 主要依赖

- `vue` `vue-router` — 框架与路由
- `element-plus` `@element-plus/icons-vue` — UI 组件
- `echarts` — 回测曲线图
- `axios` — HTTP 请求
- `vite` `@vitejs/plugin-vue` — 构建（devDependencies）
