"""仅使用交易日历和标签结束日的时间边界规则。"""
from __future__ import annotations
import pandas as pd


def label_end_dates(calendar, horizon: int) -> pd.Series:
    if horizon < 1:
        raise ValueError("预测周期必须为正交易日数")
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    return pd.Series(dates, index=dates).shift(-(horizon + 1))


def purged_mask(dates, ends, start, end, information_cutoff) -> pd.Series:
    dates = pd.Series(pd.to_datetime(dates), index=ends.index)
    return (dates.ge(pd.Timestamp(start)) & dates.le(pd.Timestamp(end))
            & ends.notna() & ends.lt(pd.Timestamp(information_cutoff)))


def purged_period(period, calendar, horizon: int, next_start):
    ends = label_end_dates(calendar, horizon)
    valid = ends.index[(ends.index >= pd.Timestamp(period[0]))
                       & (ends.index <= pd.Timestamp(period[1]))
                       & ends.lt(pd.Timestamp(next_start))]
    if len(valid) == 0:
        raise ValueError("清除跨界标签后没有训练/验证交易日")
    return str(valid[0].date()), str(valid[-1].date())
