"""
全局配置：数据范围、股票池、训练/验证/测试切分、策略参数。

只改这一个文件就能调整整套流程的行为。
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
QLIB_DATA_DIR = BASE_DIR / "qlib_data" / "cn_data"   # 生成的 Qlib 二进制数据
RAW_CACHE_DIR = BASE_DIR / "raw_cache"                # AkShare 原始行情缓存(CSV)
OUTPUT_DIR = BASE_DIR / "output"                      # 预测/选股清单/回测报告
MODEL_DIR = BASE_DIR / "models"                       # 训练好的模型
DATA_DIR = BASE_DIR / "data"                          # DuckDB 规范化真源 + 血缘

for _d in (QLIB_DATA_DIR, RAW_CACHE_DIR, OUTPUT_DIR, MODEL_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 股票池与数据范围
# ----------------------------------------------------------------------------
# ETF 池：内置宽基 + 行业主题 ETF(universe.py)，不依赖成分股接口
MARKET = "etf"             # 写入 Qlib instruments 的市场名(小写)

# 回测主基准：沪深300（以沪深300ETF代理，dump 时生成 BENCH300 不可交易实例），
# 用户关心“能否跑赢沪深300”。EQW_BENCH 为全ETF等权参考基准(衡量轮动alpha)。
BENCHMARK_SYMBOL = "BENCH300"
EQW_BENCH_SYMBOL = "BENCH"

START_DATE = "20160101"    # 抓取起始日 (YYYYMMDD)  ← 扩到2016，覆盖更长历史
END_DATE = _dt.date.today().strftime("%Y%m%d")  # 抓取结束日=今天

# 只抓前 N 只 ETF（用于限制池/快速测试）；None = 内置清单全部
MAX_STOCKS: int | None = None

# ----------------------------------------------------------------------------
# 机器学习：标签与切分
# ----------------------------------------------------------------------------
# 预测未来 N 个交易日的收益（中长线趋势 ~1 个月）。
# 标签 = 次日买入、N 日后卖出的收益率。
LABEL_HORIZON = 20

# 按日期切分（防止未来函数：训练集时间严格早于验证/测试）——仅 WALK_FORWARD=False 时用
TRAIN_PERIOD = ("2016-01-01", "2022-12-31")
VALID_PERIOD = ("2023-01-01", "2023-12-31")
TEST_PERIOD = ("2024-01-01", "2030-12-31")

# 滚动重训 / walk-forward：每个测试年用其之前的数据训练，拼接样本外预测后再回测。
# 这是真正的样本外验证（避免一次性切分的偶然性）。
WALK_FORWARD = True
WF_TRAIN_START = "2016-01-01"
WF_TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]  # 逐年滚动的样本外测试区间

# 趋势/绝对动量过滤：某标的过去 TREND_WINDOW 个交易日收益<=0 时，将其打分压到最低，
# 使 TopK 轮动避开下行趋势资产（配合池中债/金，坏年份自动轮到防守）。
USE_TREND_FILTER = True
TREND_WINDOW = 60

# ----------------------------------------------------------------------------
# 组合策略（回测）
# ----------------------------------------------------------------------------
TOPK = 6           # 持仓只数(ETF池~31只含防守资产，取~1/5做集中轮动)
N_DROP = 1         # 每次调仓最多换掉的只数(控制换手率，ETF轮动宜低)

# ETF 交易成本（免印花税，明显低于个股）
OPEN_COST = 0.0003    # 买入费率(佣金)
CLOSE_COST = 0.0003   # 卖出费率(佣金；ETF无印花税)
MIN_COST = 0.2        # 单笔最低佣金(元)
LIMIT_THRESHOLD = 0.095  # 涨跌停阈值(股票型ETF ±10%，用±9.5%近似避免涨停买/跌停卖)

# LightGBM 超参(适合小样本、稳健为主)
LGB_PARAMS = {
    "loss": "mse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
    "colsample_bytree": 0.85,
    "subsample": 0.85,
    "subsample_freq": 1,
    "min_data_in_leaf": 40,
    "lambda_l1": 1,
    "lambda_l2": 10,
}

# ----------------------------------------------------------------------------
# 数据层：源/存储/校验（Tushare 主 + AKShare 校验 + DuckDB 规范化）
# ----------------------------------------------------------------------------
# Tushare token 从环境变量注入，勿硬编码（PowerShell: $env:TUSHARE_TOKEN="xxxx"）
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
# 日线主源：tushare(需≥5000积分) | akshare(免费，自带后复权)
PRIMARY_BAR_SOURCE = "akshare"

DUCKDB_PATH = DATA_DIR / "market.duckdb"   # DuckDB 单文件库
PARQUET_DIR = DATA_DIR / "parquet"         # （预留）Parquet 导出目录

# 抽样交叉校验（Tushare 主 ↔ AKShare 校验）
VALIDATE_SAMPLE_N = 30       # 每只 ETF 抽样核对的交易日数
VALIDATE_REL_TOL = 0.005     # OHLCV 相对误差容忍(0.5%)
VALIDATE_ADJ_TOL = 0.003     # 复权一致性容忍(0.3%)

# ----------------------------------------------------------------------------
# 交易范围：只操作 ETF，且不考虑 QDII
# ----------------------------------------------------------------------------
# QDII(纳指/恒生等跨境ETF)彻底排除出可交易池、训练信号与每日决策；
# 历史数据仍保留在 DuckDB 与 all.txt 供研究。置 False 可恢复旧行为(仅溢价降权)。
EXCLUDE_QDII = True

# ----------------------------------------------------------------------------
# 每日盘中决策（src.decide，交易日 14:30 运行）
# ----------------------------------------------------------------------------
DECISION_RUN_TIME = "14:30"                     # 建议运行时刻(用当时价近似当日收盘)
POSITIONS_FILE = OUTPUT_DIR / "positions.json"  # 当前持仓状态(--apply 时更新)
MODEL_STALE_DAYS = 45                           # model.pkl 超过 N 天未重训则提示

# ----------------------------------------------------------------------------
# QDII 溢价折价过滤（策略层，叠加在信号上；EXCLUDE_QDII=True 时自然失效）
# ----------------------------------------------------------------------------
USE_PREMIUM_FILTER = True
PREMIUM_CAP = 0.03           # QDII 决策日溢价 > 3% 时降权，避免高溢价追高

# ----------------------------------------------------------------------------
# 回测滑点
# ----------------------------------------------------------------------------
SLIPPAGE_BPS = 0.0005        # 单边滑点 5bp，折入交易成本
DEAL_PRICE = "close"         # 成交价：close | vwap

# ----------------------------------------------------------------------------
# 动态 ETF 池过滤
# ----------------------------------------------------------------------------
USE_DYNAMIC_UNIVERSE = True
UNIVERSE_MIN_LIST_DAYS = 250        # 上市≥250交易日≈1年
UNIVERSE_MIN_AVG_AMOUNT = 5e7       # 近20日日均成交额≥5000万元
UNIVERSE_MIN_AUM = 2e8              # 规模(net_asset)≥2亿元
