# 前后端部署文档

Web 界面是对 CLI 管道（`src/` 四步）的可视化封装，**不改动 src 任何代码**：

```
浏览器 ←HTTP→ Vue3 前端(frontend/) ←/api→ FastAPI 后端(backend/) ─子进程→ python -m src.fetch_data / dump_qlib / train
                                                              └─读取→ output/（predictions.pkl、backtest_report.csv、train_metrics.json）
```

- 后端只做两件事：① 以子进程方式运行 README 里的管道命令（fetch/dump/train/decide）并捕获日志/指标；② 读取 `output/` 产物提供查询接口。
- 前端五个页面：总览 / 选ETF清单（对应 `python -m src.predict`）/ 每日决策（对应 `python -m src.decide` 的 `output/decision_YYYYMMDD.csv`）/ 回测分析 / 数据管道（一键 fetch→dump→train，decide 单独触发）。

---

## 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 用项目 `.venv`（已装 pyqlib/akshare/pandas） |
| Node.js | ≥ 18 | 仅构建/开发前端需要；生产运行不需要 |

---

## 2. 后端部署（FastAPI）

在项目根目录（`astock-qlib/`）操作：

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装后端依赖（装进同一个 .venv）
pip install -r backend\requirements.txt

# 启动（开发，自动重载）
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# 启动（生产）
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

接口文档：启动后访问 `http://127.0.0.1:8000/docs`（Swagger UI）。

**注意事项：**

- **必须单进程运行**（不要加 `--workers`）：管道任务状态保存在内存中。
- 同一时间只允许一个管道任务；重复触发返回 `409`。
- 每步完整日志写入 `output/pipeline_logs/<step>.log`；train 的 IC/回测指标解析后落盘 `output/train_metrics.json`。
- 子进程使用与后端相同的 Python 解释器，因此务必在 `.venv` 里启动，否则 train/fetch 会因缺依赖失败。
- train 耗时较长（特征计算 + 训练 + 回测），属正常；页面日志实时滚动。

---

## 3. 前端部署（Vue 3 + Vite）

在 `frontend/` 目录操作：

```powershell
cd frontend
npm install
```

### 开发模式（热更新，前后端分离）

```powershell
npm run dev    # 监听 5173
```

浏览器访问 `http://127.0.0.1:5173`。`vite.config.js` 已配置 `/api` 代理到 `http://127.0.0.1:8000`，需同时启动后端。

### 生产模式（推荐：单端口，后端托管前端）

```powershell
npm run build  # 产出 frontend\dist\
```

`backend/main.py` 检测到 `frontend/dist` 存在时会自动托管它。此后**只需启动后端**：

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

浏览器直接访问 `http://127.0.0.1:8000`（前端用哈希路由，无需重写规则）。

### 备选：Nginx 托管

```nginx
server {
    listen 80;
    root F:/workspace/astock-qlib/frontend/dist;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 300s;
    }
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 4. 常驻运行（Windows）

用任务计划程序或 [NSSM](https://nssm.cc) 注册为服务，示例（NSSM）：

```powershell
nssm install astock-api "F:\workspace\astock-qlib\.venv\Scripts\python.exe" "-m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
nssm set astock-api AppDirectory F:\workspace\astock-qlib
nssm start astock-api
```

---

## 5. 常见问题

- **页面显示「暂无数据」**：还没跑过训练。到「数据管道」点「一键运行」，或先按 README 跑完 CLI 四步，页面即可读取已有产物。
- **抓取失败（ProxyError/SSLEOFError）**：网络经代理不稳，多跑几次 `fetch` 增量补齐（与 CLI 行为一致）。
- **改了 `config.py` / `universe.py` 后**：重启后端生效；改了 ETF 池需重跑 fetch → dump → train。
- **跨域**：开发模式后端已放开 CORS；生产单端口托管无跨域问题。
