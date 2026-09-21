"""仅调整分组策略重选周期；每日退出和风险调整保持。"""
from copy import deepcopy
from research.goal30.group_diversification import decision as original


def period(day, frequency):
    if frequency == 'monthly': return str(day.to_period('M'))
    if frequency == 'weekly': return str(day.to_period('W-SUN'))
    if frequency == 'daily': return str(day.date())
    raise ValueError('未知重选频率')


def decision(previous, scores, prices, eligible, exiting, current, policy, variant):
    day = prices.index[-1]
    key = period(day, variant['frequency'])
    review = previous.get('review_period') != key
    state = deepcopy(previous)
    state['month'] = '__force_review__' if review else str(day.to_period('M'))
    state, targets, trace = original(state, scores, prices, eligible, exiting, current, policy, variant)
    state['review_period'] = key
    return state, targets, {**trace, 'selection_review': review, 'review_period': key}
