"""
第 3 步：训练 ETF 轮动模型并回测（升级版）。

在原有 Alpha158 -> LightGBM -> TopK 轮动的基础上，加入四项优化：
  ① 趋势/绝对动量过滤：过去 N 日收益<=0 的标的打分压到最低，TopK 避开下行资产；
  ② 跨资产防守：池中含债/金/海外 ETF（见 universe.py），坏年份自动轮到防守；
  ③ 滚动重训 / walk-forward：逐年用其之前数据训练，拼接样本外预测再回测；
  ④ 微调 TOPK/N_DROP/LABEL_HORIZON（见 config.py）。
基准对标真实沪深300（BENCH300，以沪深300ETF代理）。

用法:
    python -m src.train
"""
from __future__ import annotations

import os

# mlflow 3.x 默认禁用文件存储后端；Qlib 用它记录训练指标，此处放行
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json
import pickle

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.evaluate import risk_analysis

import config
import universe
from src.datasource import DataStore
from src.utils import load_names, picks_table, symbol_to_code


def build_dataset(train_period, valid_period, test_period) -> DatasetH:
    """构建 Alpha158 数据集：158 特征 + N 日标签；归一化只在 train 段拟合(防未来函数)。"""
    h = config.LABEL_HORIZON
    label = ([f"Ref($close, -{h + 1})/Ref($close, -1) - 1"], ["LABEL0"])

    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]
    handler = Alpha158(
        instruments=config.MARKET,
        start_time=train_period[0],
        end_time=test_period[1],
        fit_start_time=train_period[0],
        fit_end_time=train_period[1],
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        label=label,
    )
    return DatasetH(
        handler,
        segments={"train": train_period, "valid": valid_period, "test": test_period},
    )


def _predict_and_label(model: LGBModel, dataset: DatasetH):
    """返回该数据集 test 段的预测打分与真实标签(未归一化收益)。"""
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    return pred, label


def walk_forward():
    """滚动重训：每个测试年用其之前全部数据训练，拼接样本外预测。

    例：测试 2023 -> 训练 2016~2021、验证 2022、预测 2023。逐年滚动、拼接为
    一条连续的样本外预测序列，避免一次性切分的偶然性(真正的 OOS 验证)。
    """
    preds, labels = [], []
    last_model = None
    for ty in config.WF_TEST_YEARS:
        train = (config.WF_TRAIN_START, f"{ty - 2}-12-31")
        valid = (f"{ty - 1}-01-01", f"{ty - 1}-12-31")
        test = (f"{ty}-01-01", f"{ty}-12-31")
        print(f"[walk-forward] 测试 {ty}：训练 {train[0]}~{train[1]}，验证 {valid[0]}~{valid[1]}")
        ds = build_dataset(train, valid, test)
        model = LGBModel(**config.LGB_PARAMS)
        model.fit(ds)
        p, lab = _predict_and_label(model, ds)
        preds.append(p)
        labels.append(lab)
        last_model = model
    pred = pd.concat(preds)
    pred = pred[~pred.index.duplicated(keep="last")].sort_index()
    pred.name = "score"
    label = pd.concat(labels)
    label = label[~label.index.duplicated(keep="last")].sort_index()
    label.name = "label"
    return pred, label, last_model


def single_split():
    """单次切分(WALK_FORWARD=False 时)。"""
    ds = build_dataset(config.TRAIN_PERIOD, config.VALID_PERIOD, config.TEST_PERIOD)
    model = LGBModel(**config.LGB_PARAMS)
    model.fit(ds)
    pred, label = _predict_and_label(model, ds)
    pred = pred.sort_index()
    pred.name = "score"
    label.name = "label"
    return pred, label, model


def eval_ic(pred: pd.Series, label: pd.Series) -> dict:
    """计算 IC / RankIC（原始打分与真实未来收益的相关性，衡量模型质量）。"""
    df = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
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


def trend_filtered_signal(pred: pd.Series) -> pd.Series:
    """趋势/绝对动量过滤：过去 TREND_WINDOW 日收益<=0 的标的，打分压到最低，
    使 TopK 轮动避开下行趋势资产（配合池中债/金，坏年份自动轮到防守）。
    仅用过去价格，无未来函数。"""
    if not getattr(config, "USE_TREND_FILTER", False):
        return pred
    insts = sorted(set(pred.index.get_level_values("instrument")))
    d0 = pred.index.get_level_values("datetime").min()
    d1 = pred.index.get_level_values("datetime").max()
    w = config.TREND_WINDOW
    feat = D.features(insts, [f"$close/Ref($close,{w})-1"], start_time=d0, end_time=d1, freq="day")
    mom = feat.iloc[:, 0]
    # D.features 索引为 (instrument, datetime)，对齐到 pred 的 (datetime, instrument)
    mom = mom.reorder_levels(["datetime", "instrument"]).sort_index().reindex(pred.index)
    arr = pred.to_numpy(dtype=float).copy()
    penalty = float(np.nanmin(arr)) - 1000.0
    down = (mom <= 0).to_numpy(dtype=bool)  # NaN<=0 -> False(动量未知时不惩罚)
    arr[down] = penalty
    n_down = int(down.sum())
    print(f"趋势过滤：{n_down}/{len(arr)} 个(日,标的)因过去{w}日动量<=0被降权")
    return pd.Series(arr, index=pred.index, name="score")


def premium_filtered_signal(pred: pd.Series) -> pd.Series:
    """QDII 溢价过滤：决策日溢价率 > PREMIUM_CAP 的 QDII 标的打分压到最低，
    避免高溢价追高。溢价=市场收盘价/单位净值-1，取 nav_date<=当日最近净值（无未来函数）。"""
    if not getattr(config, "USE_PREMIUM_FILTER", False):
        return pred
    insts = sorted(set(pred.index.get_level_values("instrument")))
    qdii = [s for s in insts if universe.get_asset_class(symbol_to_code(str(s))) == "qdii"]
    if not qdii:
        return pred
    with DataStore() as store:
        prem = store.load_premium(qdii)
    if prem is None or prem.empty:
        print("溢价过滤：无 NAV 数据，跳过（需先拉取 fund_nav）")
        return pred
    prem = prem.reindex(pred.index)
    arr = pred.to_numpy(dtype=float).copy()
    penalty = float(np.nanmin(arr)) - 1000.0
    over = (prem > config.PREMIUM_CAP).to_numpy(dtype=bool)  # NaN>cap -> False
    arr[over] = penalty
    print(f"溢价过滤：{int(over.sum())}/{len(arr)} 个(日,标的)因 QDII 溢价>{config.PREMIUM_CAP:.1%} 被降权")
    return pd.Series(arr, index=pred.index, name="score")


def run_backtest(signal: pd.Series) -> dict:
    """TopK 轮动回测：含 ETF 成本、滑点、涨跌停限制，对标沪深300(BENCH300)与等权ETF。"""
    dates = signal.index.get_level_values("datetime").unique().sort_values()
    bt_start = dates[0]
    bt_end = dates[-2] if len(dates) >= 2 else dates[-1]

    # 滑点折入成本（Qlib Exchange 无独立滑点参数），成交价可切 close/vwap
    slippage = float(getattr(config, "SLIPPAGE_BPS", 0.0))
    deal_price = getattr(config, "DEAL_PRICE", "close")
    open_cost = config.OPEN_COST + slippage
    close_cost = config.CLOSE_COST + slippage

    strategy = TopkDropoutStrategy(signal=signal, topk=config.TOPK, n_drop=config.N_DROP)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    portfolio_dict, _ = backtest(
        start_time=bt_start,
        end_time=bt_end,
        strategy=strategy,
        executor=executor,
        account=100_000_000,
        benchmark=config.BENCHMARK_SYMBOL,  # BENCH300 = 沪深300代理
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": config.LIMIT_THRESHOLD,
            "deal_price": deal_price,
            "open_cost": open_cost,
            "close_cost": close_cost,
            "min_cost": config.MIN_COST,
        },
    )
    report = list(portfolio_dict.values())[0][0]

    strat_ret = report["return"] - report["cost"]
    excess_ret = report["return"] - report["bench"] - report["cost"]  # bench = 沪深300
    ana = risk_analysis(excess_ret, freq="day")       # 超额(vs沪深300)
    ana_abs = risk_analysis(strat_ret, freq="day")    # 策略自身

    cum_strat = (1 + strat_ret).prod() - 1
    cum_300 = (1 + report["bench"]).prod() - 1

    eqw = None
    try:
        b = D.features([config.EQW_BENCH_SYMBOL], ["$close"],
                       start_time=bt_start, end_time=bt_end, freq="day")["$close"].sort_index()
        if len(b) > 1 and b.iloc[0] > 0:
            eqw = b.iloc[-1] / b.iloc[0] - 1
    except Exception:
        pass

    tf = "开" if getattr(config, "USE_TREND_FILTER", False) else "关"
    print("\n===== 回测结果（walk-forward 样本外）=====")
    print(f"区间：{bt_start.date()} ~ {bt_end.date()}   持仓 {config.TOPK} 只ETF   趋势过滤：{tf}")
    print(f"滑点：单边 {slippage*1e4:.0f}bp（折入成本）   成交价：{deal_price}")
    print(f"策略累计收益(扣费)：  {cum_strat:.2%}")
    print(f"沪深300累计：         {cum_300:.2%}")
    if eqw is not None:
        print(f"等权ETF基准累计：     {eqw:.2%}")
    print(f"超额年化(vs沪深300)：{ana.loc['annualized_return', 'risk']:.2%}   "
          f"信息比率IR：{ana.loc['information_ratio', 'risk']:.3f}   "
          f"超额最大回撤：{ana.loc['max_drawdown', 'risk']:.2%}")
    print(f"策略自身：年化 {ana_abs.loc['annualized_return', 'risk']:.2%}   "
          f"最大回撤 {ana_abs.loc['max_drawdown', 'risk']:.2%}")

    report.to_csv(config.OUTPUT_DIR / "backtest_report.csv", encoding="utf-8-sig")
    return {
        "bt_start": str(bt_start.date()),
        "bt_end": str(bt_end.date()),
        "cum_strategy": round(float(cum_strat), 4),
        "cum_csi300": round(float(cum_300), 4),
        "excess_annual": round(float(ana.loc["annualized_return", "risk"]), 4),
        "info_ratio": round(float(ana.loc["information_ratio", "risk"]), 4),
        "excess_max_drawdown": round(float(ana.loc["max_drawdown", "risk"]), 4),
        "strategy_max_drawdown": round(float(ana_abs.loc["max_drawdown", "risk"]), 4),
        "slippage_bps": round(slippage * 1e4, 2),
        "deal_price": deal_price,
    }


def main() -> None:
    qlib.init(provider_uri=str(config.QLIB_DATA_DIR), region=REG_CN)

    if getattr(config, "WALK_FORWARD", False):
        print("模式：walk-forward 滚动重训")
        pred, label, model = walk_forward()
    else:
        print("模式：单次切分")
        pred, label, model = single_split()

    with open(config.MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 预测质量：用原始打分对真实收益，衡量模型本身(趋势过滤是策略层叠加，不计入)
    metrics = eval_ic(pred, label)
    print("\n===== 预测质量（样本外）=====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("参考：RankIC均值 > 0.03 有一定选股能力，> 0.05 较好")

    # 原始预测（未过滤）落盘，供对照脚本回测“关闭趋势过滤”版
    pred.to_pickle(config.OUTPUT_DIR / "predictions_raw.pkl")

    # 策略信号：趋势过滤 + QDII 溢价过滤（叠加），用于回测与选ETF清单
    signal = trend_filtered_signal(pred)
    signal = premium_filtered_signal(signal)
    signal.to_pickle(config.OUTPUT_DIR / "predictions.pkl")

    bt_metrics = run_backtest(signal)

    # 最新一天的选 ETF 清单（过滤后=策略实际持仓），QDII 附溢价%
    names = load_names()
    latest = signal.index.get_level_values("datetime").max()
    day = signal.xs(latest, level="datetime")
    qdii_syms = [s for s in day.index
                 if universe.get_asset_class(symbol_to_code(str(s))) == "qdii"]
    prem_map: dict = {}
    if qdii_syms:
        with DataStore() as store:
            prem_map = store.latest_premium(qdii_syms)
    table = picks_table(day, names, config.TOPK, premium=prem_map or None)
    out = config.OUTPUT_DIR / "latest_picks.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n===== 最新交易日 {latest.date()} TopK 选ETF（过滤后）=====")
    print(table.head(10).to_string(index=False))
    print(f"...(完整 {config.TOPK} 只见 {out})")

    # 溯源：把数据版本(来自 dump 的 run_meta) + IC + 回测指标写回 run_meta.json
    _write_run_meta(metrics, bt_metrics)


def _write_run_meta(ic_metrics: dict, bt_metrics: dict) -> None:
    """将训练/回测结果并入 dump 生成的 run_meta.json，形成“模型←数据版本”溯源链。"""
    path = config.OUTPUT_DIR / "run_meta.json"
    meta = {}
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["trained_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["ic_metrics"] = ic_metrics
    meta["backtest"] = bt_metrics
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
