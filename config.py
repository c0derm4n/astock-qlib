"""
全局配置：数据范围、股票池、训练/验证/测试切分、策略参数。

只改这一个文件就能调整整套流程的行为。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
QLIB_DATA_DIR = BASE_DIR / "qlib_data" / "cn_data"   # 生成的 Qlib 二进制数据
RAW_CACHE_DIR = BASE_DIR / "raw_cache"                # AkShare 原始行情缓存(CSV)
OUTPUT_DIR = BASE_DIR / "output"                      # 预测/选股清单/回测报告
MODEL_DIR = BASE_DIR / "models"                       # 训练好的模型

for _d in (QLIB_DATA_DIR, RAW_CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 股票池与数据范围
# ----------------------------------------------------------------------------
# ETF 池：内置宽基 + 行业主题 ETF(universe.py)，不依赖成分股接口
MARKET = "etf"             # 写入 Qlib instruments 的市场名(小写)

# 基准：由 ETF 池合成的「等权指数」(无需联网取指数)，用于回测超额收益
# 代表“均衡持有全部 ETF”的基准，策略能否跑赢它=行业/风格轮动能力的直接体现
BENCHMARK_SYMBOL = "BENCH"

START_DATE = "20180101"    # 抓取起始日 (YYYYMMDD)
END_DATE = _dt.date.today().strftime("%Y%m%d")  # 抓取结束日=今天

# 只抓前 N 只 ETF（用于限制池/快速测试）；None = 内置清单全部
MAX_STOCKS: int | None = None

# ----------------------------------------------------------------------------
# 机器学习：标签与切分
# ----------------------------------------------------------------------------
# 预测未来 N 个交易日的收益（中长线趋势 ~1 个月）。
# 标签 = 次日买入、N 日后卖出的收益率。
LABEL_HORIZON = 20

# 按日期切分（防止未来函数：训练集时间严格早于验证/测试）
TRAIN_PERIOD = ("2018-01-01", "2022-12-31")
VALID_PERIOD = ("2023-01-01", "2023-12-31")
TEST_PERIOD = ("2024-01-01", "2030-12-31")

# ----------------------------------------------------------------------------
# 组合策略（回测）
# ----------------------------------------------------------------------------
TOPK = 5           # 持仓只数(ETF池~27只，取~1/5做集中轮动)
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
