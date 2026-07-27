# A股 ETF 轮动系统（Qlib + LightGBM）

用机器学习给一篮子 A股 ETF **每日打分排名、做行业/风格轮动**的最小可用系统。
（本项目由"买个股"升级为**只买 ETF**：单只 ETF 已分散、无个股暴雷、行业动量更持续，且**免印花税**、成本更低，更契合 TopK 排序轮动。）

- **标的**：内置**宽基 + 行业主题 + 跨资产防守 ETF**（沪深300/500/1000/50/创业板/科创/红利 + 证券/银行/医药/消费/芯片/军工/光伏/新能源车/有色/煤炭… + 十年国债/黄金/纳指/恒生，见 `universe.py`）
- **特征**：微软 Qlib 内置的 **Alpha158**——158 个技术指标（K线形态、均线、动量ROC、波动率、量价相关性、类RSI等）
- **模型**：LightGBM 预测未来 20 个交易日收益，做横截面排序轮动（中长线趋势）
- **策略**：TopK 轮动 + **趋势过滤**（过去60日收益≤0的资产降权，避开下行）+ **跨资产防守**（池含债/金/纳指/恒生，坏年份自动轮到防守）
- **验证**：**walk-forward 滚动重训**（逐年样本外，2021–2026），回测计入 ETF 成本/涨跌停
- **输出**：每个交易日一份 TopK 选 ETF 清单（代码+名称+打分）
- **对标**：真实**沪深300**（BENCH300，以沪深300ETF代理）＋全ETF等权参考

> ⚠️ 本项目是**决策支持工具**，输出为模型打分排名，**不是投资建议**。务必先模拟盘验证、结合宏观与风控，盈亏自负。

---

## 目录结构

```
astock-qlib/
├── config.py          # ★ 全局配置（ETF池/日期/数据源/校验/滑点/动态池/超参）
├── universe.py        # 内置 ETF 候选池（代码->名称）+ 资产类别
├── src/
│   ├── datasource/    # 数据访问层：Tushare主源 + AKShare校验 + DuckDB 存储
│   │   ├── store.py           # DuckDB 规范化真源 + 血缘 + 后复权/溢价派生
│   │   ├── tushare_source.py  # fund_daily/fund_adj/fund_nav/etf_basic
│   │   └── akshare_source.py  # fund_etf_hist_em（校验/补数）
│   ├── fetch_data.py  # 第1步：抓数据 -> DuckDB（含抽样交叉校验）
│   ├── dump_qlib.py   # 第2步：DuckDB(后复权) -> Qlib 二进制 + 动态池 + 溯源
│   ├── train.py       # 第3步：Alpha158->LGB->IC+回测(滑点)+溢价过滤+选ETF
│   ├── predict.py     # 第4步：输出某交易日 TopK 选 ETF 清单
│   ├── universe_filter.py  # 动态 ETF 池过滤（上市时长/流动性/规模）
│   ├── data_report.py # 数据血缘报告（拉取/校验汇总）
│   └── utils.py       # 名称映射、代码转换、清单格式化
├── data/              # DuckDB 规范化真源 + 血缘（gitignore）
├── raw_cache/         # 旧版原始行情缓存（只读兼容，gitignore）
├── qlib_data/cn_data/ # 生成的 Qlib 数据（gitignore）
├── models/            # 训练好的模型（gitignore）
├── output/            # 预测/选ETF清单/回测报告/血缘/溯源（gitignore）
└── requirements.txt
```

---

## 快速开始

环境已在 `.venv` 内装好（Python 3.10 + pyqlib 0.9.7 + tushare + akshare + duckdb）。所有命令在 `astock-qlib/` 目录下执行。

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 配置 Tushare token（主源，需 ≥5000 积分；仅当前会话有效）
$env:TUSHARE_TOKEN="你的token"
# 积分不足时可临时改用免费兜底：在 config.py 设 PRIMARY_BAR_SOURCE="akshare"

# 第1步：抓取 ETF 数据到 DuckDB（Tushare 主 + AKShare 抽样校验；增量）
python -m src.fetch_data
python -m src.fetch_data --force         # 全量重拉
python -m src.fetch_data --no-validate   # 跳过 AKShare 抽样校验（更快）

# 第2步：DuckDB(后复权) -> Qlib 二进制 + 动态池过滤 + 数据溯源
python -m src.dump_qlib

# 第3步：训练 + 回测(含滑点) + QDII溢价过滤 + 生成最新选 ETF 清单
python -m src.train

# 第4步：查看/复盘某日选 ETF 清单
python -m src.predict                    # 最新交易日 TopK
python -m src.predict --date 2025-06-30  # 指定历史某日复盘
python -m src.predict --topk 10          # 只看前10

# （可选）查看数据血缘报告：拉取/校验汇总
python -m src.data_report
```

**日常使用**：中长线约每月调仓一次。每次想更新时，依次跑 `fetch_data → dump_qlib → train`（train 会自动重训并输出最新清单），再用 `predict` 查看。

---

## 如何看结果

**1. 预测质量（train 输出的 IC/RankIC）**

| 指标 | 含义 | 参考 |
|------|------|------|
| RankIC均值 | 打分与未来收益的排序相关性 | >0.03 有一定轮动力，>0.05 较好 |
| RankICIR | RankIC 的稳定性（均值/标准差） | 越高越稳定 |

**2. 回测（train 输出 + `output/backtest_report.csv`）**
- 策略累计收益 vs **等权 ETF 基准**：衡量轮动是否带来超额（选对板块的能力）
- 策略累计收益 vs **买入持有沪深300ETF**：衡量能否跑赢"躺平持有大盘"
- 超额年化、信息比率IR：>0 才有超额价值；最大回撤：能承受的最大浮亏

**3. 选 ETF 清单（`output/latest_picks.csv`）**
- 按模型打分从高到低排序的 TopK 只 ETF

---

## ⚠️ 关于结果（务必阅读，坦诚说）

ETF 轮动比个股选股**更契合本管道**，但仍**不是稳赢的银弹**：

1. **信号仍偏弱**：纯技术指标的 RankIC 常只有 0.01~0.05，A股噪声大，仍有过拟合风险；
2. **轮动会被打脸**：动量轮动在震荡/风格急切换时会"追高杀跌"来回挨打；
3. **QDII 有偏差**：纳指/恒生 ETF 存在溢价折价与汇率/时差影响，回测按 A股 ±10% 近似处理。

本项目价值在于给你一套**正确、可扩展的 ETF 轮动管道**，而非一个现成的赚钱策略。

### 已内置的稳健化改进
1. **趋势/绝对动量过滤**：过去 `TREND_WINDOW`(默认60)日收益≤0 的标的打分压到最低，TopK 自动避开下行资产；
2. **跨资产防守**：池含十年国债/黄金/纳指/恒生，坏年份轮到防守资产（配合趋势过滤=类空仓）；
3. **walk-forward 滚动重训**：逐年用其之前数据训练、拼接样本外预测(2021–2026)再回测，是真正的 OOS 验证；
4. **对标真实沪深300**：超额、信息比率、回撤都相对沪深300 计算。

### 可继续提升
- 抓全/加长历史（多跑 `fetch_data`）；换/调模型（`config.LGB_PARAMS`）；加基本面或宏观因子；对 QDII 用更贴合的交易规则。

---

## 常用配置（`config.py`）

| 配置 | 说明 |
|------|------|
| `MARKET` | Qlib instruments 市场名（ETF 版为 `"etf"`） |
| `START_DATE` | 数据起始日（默认 20160101；部分 ETF 上市晚，早期样本少属正常） |
| `WALK_FORWARD` / `WF_TEST_YEARS` | 是否滚动重训 / 逐年样本外测试区间（默认 2021–2026） |
| `USE_TREND_FILTER` / `TREND_WINDOW` | 趋势过滤开关 / 绝对动量回看天数（默认 开 / 60） |
| `BENCHMARK_SYMBOL` | 回测主基准（默认 `BENCH300`=沪深300代理）；`EQW_BENCH_SYMBOL`=等权参考 |
| `LABEL_HORIZON` | 预测未来多少交易日的收益（默认20≈1个月） |
| `TRAIN/VALID/TEST_PERIOD` | 单次切分日期（仅 `WALK_FORWARD=False` 时用） |
| `TOPK` / `N_DROP` | 持仓只数 / 每次最多换手只数（默认 6/1） |
| `OPEN_COST/CLOSE_COST` | 买入/卖出费率（ETF **免印花税**，默认 0.0003） |
| `LGB_PARAMS` | LightGBM 超参 |
| `PRIMARY_BAR_SOURCE` | 日线主源：`tushare`(默认) / `akshare`(免费兜底) |
| `SLIPPAGE_BPS` / `DEAL_PRICE` | 回测单边滑点(默认5bp，折入成本) / 成交价(close或vwap) |
| `USE_PREMIUM_FILTER` / `PREMIUM_CAP` | QDII 溢价过滤开关 / 决策日溢价上限(默认3%) |
| `USE_DYNAMIC_UNIVERSE` | 动态池过滤开关(上市时长/流动性/规模) |
| `VALIDATE_SAMPLE_N` | 每只 ETF 抽样交叉校验的交易日数 |

---

## 数据说明

- **主源 Tushare**：`fund_daily`(不复权日线) + `fund_adj`(复权因子) + `fund_nav`(净值/规模) + `etf_basic`(元数据)；需 `TUSHARE_TOKEN`（≥5000 积分，`etf_basic` 需 8000，不足自动降级 `fund_basic`）。
- **校验/补数/备用主源 AKShare**：优先东财 `fund_etf_hist_em`（不复权），**东财不可达时自动降级新浪 `fund_etf_hist_sina`（前复权，此时 adj_factor=1）**；作为 Tushare 无权限时的免费主源（`PRIMARY_BAR_SOURCE="akshare"`）。
- **后复权**：dump 时 `hfq = 原价 × adj_factor`（历史稳定、回测可复现），分红自动并入全收益。
- **QDII 溢价折价**：由 `fund_nav` 计算 `溢价=收盘价/单位净值-1`，决策日溢价 > `PREMIUM_CAP` 的 QDII 降权（`config.USE_PREMIUM_FILTER`）。
- **本地存储**：DuckDB 规范化真源（`data/market.duckdb`），全量记录来源/拉取时间/数据版本/校验结果（血缘）；再 dump 成 Qlib 二进制。
- 回测主基准仍为**沪深300**（`BENCH300`）+ 全ETF等权 `BENCH`；单只失败跳过、增量续拉。

## 已知限制
- 部分行业 ETF 于 2019–2021 才上市，早期训练样本较少（Qlib 按各标的自身起止日处理）；
- 主基准沪深300以沪深300ETF代理(与指数略有跟踪误差)；QDII(纳指/恒生)溢价折价已由 NAV 计算并在决策日过滤，但回测撮合仍按 A股 ±10% 规则近似；
- Qlib 训练时会打印若干 `CatBoost/XGB/PyTorch skipped` 与 gym 警告，**均无害**（未安装的可选模型）。
