# A股 ETF 轮动系统（Qlib + LightGBM）

用机器学习给一篮子 A股 ETF **每日打分排名、做行业/风格轮动**的最小可用系统。
（本项目由"买个股"升级为**只买 ETF**：单只 ETF 已分散、无个股暴雷、行业动量更持续，且**免印花税**、成本更低，更契合 TopK 排序轮动。）

- **标的**：内置**宽基 + 行业主题 ETF**（沪深300/500/1000/50/创业板/科创/红利/深100 + 证券/银行/医药/消费/芯片/军工/光伏/新能源车/有色/煤炭…，见 `universe.py`）
- **特征**：微软 Qlib 内置的 **Alpha158**——158 个技术指标（K线形态、均线、动量ROC、波动率、量价相关性、类RSI等）
- **模型**：LightGBM 预测未来 20 个交易日收益，做横截面排序轮动（中长线趋势）
- **输出**：每个交易日一份 TopK 选 ETF 清单（代码+名称+打分）
- **回测**：TopK 组合策略，计入 ETF 交易成本、涨跌停限制，对标「等权 ETF 基准」与「买入持有沪深300ETF」

> ⚠️ 本项目是**决策支持工具**，输出为模型打分排名，**不是投资建议**。务必先模拟盘验证、结合宏观与风控，盈亏自负。

---

## 目录结构

```
astock-qlib/
├── config.py          # ★ 全局配置（ETF池/日期/切分/成本/超参），只改这里
├── universe.py        # 内置 ETF 池（代码->名称）
├── src/
│   ├── fetch_data.py  # 第1步：AkShare 抓 ETF 行情 -> raw_cache/*.csv
│   ├── dump_qlib.py   # 第2步：CSV -> Qlib 二进制数据 + 合成等权基准
│   ├── train.py       # 第3步：Alpha158 -> LightGBM -> IC评估 + 回测 + 选ETF
│   ├── predict.py     # 第4步：输出某交易日 TopK 选 ETF 清单
│   └── utils.py       # 名称映射、代码转换、清单格式化
├── raw_cache/         # 原始行情缓存（gitignore）
├── qlib_data/cn_data/ # 生成的 Qlib 数据（gitignore）
├── models/            # 训练好的模型（gitignore）
├── output/            # 预测结果、选ETF清单、回测报告（gitignore）
└── requirements.txt
```

---

## 快速开始

环境已在 `.venv` 内装好（Python 3.10 + pyqlib 0.9.7 + akshare）。所有命令在 `astock-qlib/` 目录下执行。

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 第1步：抓取 ETF 行情（增量，已缓存的会跳过；失败自动重试）
python -m src.fetch_data
# 网络/代理不稳时，多跑几遍即可增量补齐（只抓还没缓存的）

# 第2步：转成 Qlib 数据格式（自动忽略非当前 ETF 池的历史遗留缓存）
python -m src.dump_qlib

# 第3步：训练 + 回测 + 生成最新选 ETF 清单
python -m src.train

# 第4步：查看/复盘某日选 ETF 清单
python -m src.predict                    # 最新交易日 TopK
python -m src.predict --date 2025-06-30  # 指定历史某日复盘
python -m src.predict --topk 10          # 只看前10
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

1. **信号仍偏弱**：纯技术指标的 RankIC 常只有 0.01~0.05，A股噪声大；~27 只标的 + 158 个因子仍有过拟合风险；
2. **轮动会被打脸**：动量轮动在震荡/风格急切换时会"追高杀跌"来回挨打；
3. **风格错配**：若测试期是红利/价值/大盘领涨，纯动量轮动器可能仍跑不过简单持有红利/300。

**真正结构性提升抗跌能力的，是加入趋势过滤或跨资产防守（见下）。** 本项目价值在于给你一套**正确、可扩展的 ETF 轮动管道**，而非一个现成的赚钱策略。

### 如何提升（按性价比排序）
1. **抓全 ETF 池**：网络好时多跑几遍 `fetch_data` 增量补齐；标的越全、横截面轮动越有效；
2. **加趋势/绝对动量过滤**（对回撤帮助最大、最不易过拟合）：当排名第一的 ETF 自身动量都为负时，不满仓、转持债券 ETF 或现金；
3. **加跨资产防守**：把债券（国债/信用债）、黄金、海外（纳指/恒生）ETF 放进池，坏年份能轮动到防守资产；
4. **调参与标签**：调 `TOPK`/`N_DROP`/`LABEL_HORIZON`、`config.LGB_PARAMS`；
5. **滚动重训**：定期用最新数据重训，做真正的样本外滚动验证。

---

## 常用配置（`config.py`）

| 配置 | 说明 |
|------|------|
| `MARKET` | Qlib instruments 市场名（ETF 版为 `"etf"`） |
| `MAX_STOCKS` | 限制 ETF 池大小；`None`=用内置清单全部 |
| `START_DATE` | 数据起始日（部分行业 ETF 上市晚，早期样本少属正常） |
| `LABEL_HORIZON` | 预测未来多少交易日的收益（默认20≈1个月） |
| `TRAIN/VALID/TEST_PERIOD` | 训练/验证/测试的日期切分（严格防未来函数） |
| `TOPK` / `N_DROP` | 持仓只数 / 每次最多换手只数（ETF 池小，默认 5/1） |
| `OPEN_COST/CLOSE_COST` | 买入/卖出费率（ETF **免印花税**，默认 0.0003） |
| `LGB_PARAMS` | LightGBM 超参 |

---

## 数据说明

- 数据源：**AkShare** 的 `fund_etf_hist_em`（东方财富 ETF 接口），前复权日线；
- 基准为**股票池等权指数**（`BENCH`，dump 时合成），非真实指数；train 另会对标**买入持有沪深300ETF**；
- 抓取带自动重试；单只失败会跳过、不影响整体；已缓存的会跳过（增量）；
- **若网络经代理访问不稳**（偶发 `ProxyError`/`SSLEOFError`），多跑几遍 `fetch_data` 增量补齐即可。

## 已知限制
- 部分行业 ETF 于 2019–2021 才上市，早期训练样本较少（Qlib 按各标的自身起止日处理）；
- 合成等权基准非真实指数；已额外对标买入持有 300ETF 作为真实市场参照；
- Qlib 训练时会打印若干 `CatBoost/XGB/PyTorch skipped` 与 gym 警告，**均无害**（未安装的可选模型）。
