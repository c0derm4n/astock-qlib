"""重新训练只看30只ETF的季度滚动模型；严格隔离未成熟标签。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from research.ex_nasdaq.core import market30, SYMBOLS
from research.ex_nasdaq.explore import BASE, OUT, write_json
from src.models import sample_frame, ranked_labels


def build():
    source = BASE/'output/runs/20260912-220345-f20ea3f5'
    market = market30(source)
    x, y = sample_frame(market)
    dates = x.index.get_level_values('datetime')
    ends = market.label_ends.reindex(x.index)
    directory = OUT/'models'/pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'hypothesis': '仅用已成熟收益标签学习ETF横截面排序，季度重训适应不同市场环境。',
                'training_years': 3, 'min_rows': 300, 'min_dates': 60, 'label_horizon': 20,
                'models': ['ridge', 'lightgbm'], 'ridge_alpha': 100,
                'lightgbm': {'n_estimators': 150, 'learning_rate': .03, 'num_leaves': 7, 'max_depth': 3,
                             'min_child_samples': 100, 'reg_alpha': 1., 'reg_lambda': 10., 'random_state': 20260912,
                             'n_jobs': 1, 'verbosity': -1, 'deterministic': True, 'force_col_wise': True},
                'symbols': sorted(SYMBOLS), 'research_only': True, 'selection': 'fixed_hyperparameters_no_early_stopping'}
    write_json(directory/'protocol.json', protocol)
    predictions = {kind: [] for kind in protocol['models']}
    folds = []
    for quarter in pd.period_range(market.calendar[0], market.calendar[-1], freq='Q'):
        cutoff = quarter.start_time
        end = (quarter+1).start_time
        train = (dates >= cutoff-pd.DateOffset(years=3)) & (dates < cutoff) & ends.lt(cutoff) & y.notna()
        test = (dates >= cutoff) & (dates < end)
        if not test.any() or train.sum() < 300 or dates[train].nunique() < 60:
            continue
        fold = {'quarter': str(quarter), 'prediction_start': str(cutoff.date()), 'train_rows': int(train.sum()),
                'train_signal_end': str(dates[train].max().date()), 'train_label_end': str(ends[train].max().date())}
        if not ends[train].lt(cutoff).all():
            raise ValueError('训练混入未成熟标签')
        prep = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', RobustScaler())])
        train_x = prep.fit_transform(x.loc[train])
        test_x = prep.transform(x.loc[test])
        target = ranked_labels(y.loc[train])
        for kind in protocol['models']:
            model = Ridge(alpha=100.) if kind == 'ridge' else LGBMRegressor(**protocol['lightgbm'])
            model.fit(train_x, target)
            predictions[kind].append(pd.Series(model.predict(test_x), index=x.loc[test].index, name='score'))
        folds.append(fold)
        if quarter.quarter == 4:
            print(f'{quarter.year}完成，最近训练标签截止{fold["train_label_end"]}', flush=True)
    for kind, parts in predictions.items():
        scores = pd.concat(parts).unstack('instrument').reindex(index=market.calendar, columns=sorted(SYMBOLS))
        scores.to_parquet(directory/f'{kind}.parquet')
    write_json(directory/'folds.json', folds)
    files = [Path(__file__), BASE/'src/models.py', BASE/'src/market.py', BASE/'src/temporal.py',
             BASE/'src/universe_filter.py', BASE/'research/ex_nasdaq/core.py']
    files += [source/name for name in ['market_snapshot.parquet','nav_snapshot.parquet','metadata_snapshot.parquet']]
    write_json(directory/'manifest.json', {'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
               'source_hashes': {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               'artifact_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()},
               'folds': len(folds), 'all_label_ends_before_predictions': all(f['train_label_end'] < f['prediction_start'] for f in folds)})
    write_json(OUT/'models/latest.json', {'directory': str(directory)})
    print(f'模型完成：{directory}', flush=True)
    return directory


if __name__ == '__main__':
    build()
