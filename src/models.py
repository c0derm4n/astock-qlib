"""低复杂度候选、训练期拟合的预处理器与可独立复用的生产模型。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.temporal import purged_mask


def learn_bundle(features, labels, kind, seed, valid_features=None, valid_labels=None, rounds=None):
    prep = Pipeline([("fill", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", RobustScaler())])
    x = prep.fit_transform(features)
    if kind == "ridge":
        model = Ridge(alpha=100.)
        model.fit(x, labels)
        best = None
    elif kind == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=rounds or 300, learning_rate=.03, num_leaves=7,
                                  max_depth=3, min_child_samples=100, reg_alpha=1., reg_lambda=10.,
                                  random_state=seed, n_jobs=4, verbosity=-1, deterministic=True,
                                  force_col_wise=True)
        kwargs = {}
        if valid_features is not None and len(valid_features) and rounds is None:
            kwargs = {"eval_set": [(prep.transform(valid_features), valid_labels)],
                      "callbacks": [lgb.early_stopping(30, verbose=False)]}
        model.fit(x, labels, **kwargs)
        best = int(model.best_iteration_ or model.n_estimators)
    else:
        raise ValueError(f"未知模型 {kind}")
    return {"kind": kind, "preprocessor": prep, "model": model,
            "features": list(features.columns), "best_iteration": best}


def predict_bundle(bundle, features):
    if list(features.columns) != bundle["features"]:
        raise ValueError("特征版本与生产模型不一致")
    return pd.Series(bundle["model"].predict(bundle["preprocessor"].transform(features)), index=features.index, name="score")


def sample_frame(market):
    eligibility = market.eligibility.set_index(["date", "symbol"])["eligible"]
    eligibility.index.names = market.features.index.names
    mask = eligibility.reindex(market.features.index).fillna(False) & market.features["momentum_120"].notna()
    return market.features.loc[mask], market.labels.reindex(market.features.index).loc[mask]


def ranked_labels(labels):
    return labels.groupby(level="datetime").rank(pct=True) - .5


def walk_forward_models(market, protocol):
    x, y = sample_frame(market)
    ends = market.label_ends.reindex(x.index)
    dates = x.index.get_level_values("datetime")
    predictions = {k: [] for k in protocol["candidate_models"]}
    folds = []
    for year in range(pd.Timestamp(protocol["research_start"]).year, market.calendar[-1].year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        valid_start = test_start - pd.DateOffset(years=1)
        train_mask = purged_mask(dates, ends, dates.min(), valid_start - pd.Timedelta(days=1), valid_start) & y.notna()
        valid_mask = purged_mask(dates, ends, valid_start, test_start - pd.Timedelta(days=1), test_start) & y.notna()
        test_mask = (dates >= test_start) & (dates < test_start + pd.DateOffset(years=1))
        if not test_mask.any():
            continue
        if train_mask.sum() < 200 or valid_mask.sum() < 50:
            raise ValueError(f"{year}清除跨界标签后训练/验证样本不足")
        fold = {"year": year, "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum()),
                "train_last_label_end": str(ends[train_mask].max().date()),
                "valid_last_label_end": str(ends[valid_mask].max().date()), "models": {}}
        for kind in protocol["candidate_models"]:
            if kind == "momentum":
                prediction = x.loc[test_mask, "momentum_60"].rename("score")
            else:
                bundle = learn_bundle(x[train_mask], ranked_labels(y[train_mask]), kind, protocol["seed"],
                                      x[valid_mask], ranked_labels(y[valid_mask]))
                prediction = predict_bundle(bundle, x.loc[test_mask])
                fold["models"][kind] = {"best_iteration": bundle["best_iteration"]}
            predictions[kind].append(prediction)
        folds.append(fold)
        print(f"滚动窗口 {year} 完成：train={train_mask.sum()} valid={valid_mask.sum()} test={test_mask.sum()}", flush=True)
    return {kind: pd.concat(parts).sort_index() for kind, parts in predictions.items()}, folds


def fit_production(market, protocol):
    x, y = sample_frame(market)
    dates = x.index.get_level_values("datetime")
    ends = market.label_ends.reindex(x.index)
    cutoff = market.calendar[-1] + pd.Timedelta(days=1)
    valid_start = market.calendar[-1] - pd.DateOffset(years=1)
    train = purged_mask(dates, ends, dates.min(), valid_start - pd.Timedelta(days=1), valid_start) & y.notna()
    valid = purged_mask(dates, ends, valid_start, market.calendar[-1], cutoff) & y.notna()
    if train.sum() < 200 or valid.sum() < 50:
        raise ValueError("生产训练清除未成熟标签后样本不足")
    selected = learn_bundle(x[train], ranked_labels(y[train]), protocol["primary_model"], protocol["seed"], x[valid], ranked_labels(y[valid]))
    mature = y.notna() & ends.notna() & ends.lt(cutoff)
    bundle = learn_bundle(x[mature], ranked_labels(y[mature]), protocol["primary_model"], protocol["seed"], rounds=selected["best_iteration"])
    bundle.update({"trained_as_of": str(market.calendar[-1].date()),
                   "train_signal_end": str(dates[mature].max().date()),
                   "train_label_end": str(ends[mature].max().date()), "train_rows": int(mature.sum()),
                   "data_fingerprint": market.fingerprint, "execution": protocol["execution"]})
    return bundle
