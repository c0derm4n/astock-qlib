"""
第 3 步：训练 LightGBM 选股模型并回测。

流程：
  Alpha158(158个技术指标特征) -> LightGBM 预测未来N日收益 ->
  IC/RankIC 评估预测质量 -> TopK 组合回测(含ETF成本、对标等权ETF与300ETF) ->
  保存模型/预测/最新选ETF清单

用法:
    python -m src.train
"""
from __future__ import annotations

import os

# mlflow 3.x 默认禁用文件存储后端；Qlib 用它记录训练指标，此处放行
# 必须在导入 qlib/mlflow 之前设置
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pickle

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.evaluate import risk_analysis

import config
from src.utils import load_names, picks_table


def build_dataset() -> DatasetH:
    """构建 Alpha158 数据集：特征158个 + 自定义N日标签。"""
    # 标签：次日买入、N日后卖出的收益率（中长线）
    h = config.LABEL_HORIZON
    label = ([f"Ref($close, -{h + 1})/Ref($close, -1) - 1"], ["LABEL0"])

    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = [
        {"class": "DropnaLabel"},
        # 截面排序归一化：把每日标签变成横截面排名，适合选股(相对强弱)
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]

    handler = Alpha158(
        instruments=config.MARKET,
        start_time=config.TRAIN_PERIOD[0],
        end_time=config.TEST_PERIOD[1],
        fit_start_time=config.TRAIN_PERIOD[0],
        fit_end_time=config.TRAIN_PERIOD[1],
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        label=label,
    )
    return DatasetH(
        handler,
        segments={
            "train": config.TRAIN_PERIOD,
            "valid": config.VALID_PERIOD,
            "test": config.TEST_PERIOD,
        },
    )


def eval_ic(pred: pd.Series, dataset: DatasetH) -> dict:
    """计算 IC / RankIC（预测分数与真实未来收益的相关性）。"""
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    label = label.iloc[:, 0].rename("label")
    df = pd.concat([pred.rename("score"), label], axis=1).dropna()
    if df.empty:
        return {}
    g = df.groupby(level="datetime")
    ic = g.apply(lambda x: x["score"].corr(x["label"]))
    ric = g.apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    return {
        "IC均值": round(ic.mean(), 4),
        "ICIR": round(ic.mean() / ic.std(), 4) if ic.std() else float("nan"),
        "RankIC均值": round(ric.mean(), 4),
        "RankICIR": round(ric.mean() / ric.std(), 4) if ric.std() else float("nan"),
    }


def run_backtest(pred: pd.Series) -> None:
    """TopK 组合回测：含ETF交易成本、涨跌停限制，对标等权ETF基准与买入持有300ETF。"""
    dates = pred.index.get_level_values("datetime").unique().sort_values()
    bt_start = dates[0]
    # 结束日回退一天：执行引擎需要下一交易日成交，避免越界
    bt_end = dates[-2] if len(dates) >= 2 else dates[-1]

    strategy = TopkDropoutStrategy(signal=pred, topk=config.TOPK, n_drop=config.N_DROP)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    portfolio_dict, _ = backtest(
        start_time=bt_start,
        end_time=bt_end,
        strategy=strategy,
        executor=executor,
        account=100_000_000,
        benchmark=config.BENCHMARK_SYMBOL,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": config.LIMIT_THRESHOLD,
            "deal_price": "close",
            "open_cost": config.OPEN_COST,
            "close_cost": config.CLOSE_COST,
            "min_cost": config.MIN_COST,
        },
    )
    report = list(portfolio_dict.values())[0][0]

    strat_ret = report["return"] - report["cost"]
    excess_ret = report["return"] - report["bench"] - report["cost"]
    ana = risk_analysis(excess_ret, freq="day")

    cum_strat = (1 + strat_ret).prod() - 1
    cum_bench = (1 + report["bench"]).prod() - 1

    # 额外对标：买入持有沪深300ETF(真实市场基准，衡量能否跑赢“躺平”)
    hs300 = None
    try:
        from qlib.data import D
        c300 = D.features(["SH510300"], ["$close"], start_time=bt_start,
                          end_time=bt_end, freq="day")["$close"].sort_index()
        if len(c300) > 1 and c300.iloc[0] > 0:
            hs300 = c300.iloc[-1] / c300.iloc[0] - 1
    except Exception:
        pass

    print("\n===== 回测结果（测试集）=====")
    print(f"区间：{bt_start.date()} ~ {bt_end.date()}   持仓 {config.TOPK} 只ETF")
    print(f"策略累计收益(扣费)：  {cum_strat:.2%}")
    print(f"等权ETF基准累计：     {cum_bench:.2%}")
    if hs300 is not None:
        print(f"买入持有沪深300ETF：  {hs300:.2%}")
    print(f"超额年化(vs等权ETF)：{ana.loc['annualized_return', 'risk']:.2%}   "
          f"信息比率IR：{ana.loc['information_ratio', 'risk']:.3f}   "
          f"最大回撤：{ana.loc['max_drawdown', 'risk']:.2%}")

    report.to_csv(config.OUTPUT_DIR / "backtest_report.csv", encoding="utf-8-sig")


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)

    print("构建数据集(计算 Alpha158 特征)...")
    dataset = build_dataset()

    print("训练 LightGBM...")
    model = LGBModel(**config.LGB_PARAMS)
    model.fit(dataset)
    with open(config.MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("预测测试集...")
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred.name = "score"
    pred.to_pickle(config.OUTPUT_DIR / "predictions.pkl")

    metrics = eval_ic(pred, dataset)
    print("\n===== 预测质量 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("参考：RankIC均值 > 0.03 即有一定选股能力，> 0.05 较好")

    run_backtest(pred)

    # 最新一天的选股清单
    names = load_names()
    latest = pred.index.get_level_values("datetime").max()
    table = picks_table(pred.xs(latest, level="datetime"), names, config.TOPK)
    out = config.OUTPUT_DIR / "latest_picks.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n===== 最新交易日 {latest.date()} TopK 选ETF =====")
    print(table.head(10).to_string(index=False))
    print(f"...(完整 {config.TOPK} 只见 {out})")


if __name__ == "__main__":
    main()
