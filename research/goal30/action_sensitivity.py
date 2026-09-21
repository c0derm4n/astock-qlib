"""只修正内存中的折算比例做敏感性回放，不更改被冻结的历史输入。"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from backend.artifacts import write_json_atomic
from research.ex_nasdaq.core import market30, execute
from research.ex_nasdaq.replay import load_selected
from research.goal30.audit import BASE, OUT, SELECTED, comparison
from research.sp500.validate import relative_metrics, ledger_audit


def adjusted_split_bars(bars, symbol, events):
    result = bars.copy()
    changes = []
    dates = [pd.Timestamp(e['effective_date']) for e in events]
    if dates != sorted(set(dates)):
        raise ValueError('折算事件须按生效日期严格递增')
    for event, date in zip(events, dates):
        part = result[result.symbol.eq(symbol)].sort_values('date')
        before = part[part.date.lt(date)]
        after = part[part.date.eq(date)]
        ratio = float(event['ratio'])
        if before.empty or len(after) != 1 or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('折算日期、比例或前一日行情无效')
        observed = float(after.adj_factor.iloc[0] / before.adj_factor.iloc[-1])
        multiplier = ratio / observed
        mask = result.symbol.eq(symbol) & result.date.ge(date)
        result.loc[mask, 'adj_factor'] *= multiplier
        changes.append({'date': str(date.date()), 'old_ratio': observed, 'tested_ratio': ratio,
                        'relative_multiplier': multiplier})
    for field in ('open', 'close'):
        result['adj_' + field] = result[field] * result.adj_factor
    return result, changes


def run():
    protocol_path = Path(__file__).with_name('action_sensitivity_protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    selected, _ = load_selected(SELECTED)
    market = market30(BASE / selected['source'])
    if str(market.calendar[-1].date()) != protocol['end']:
        raise ValueError('行情截止日与事前登记不一致')
    pointer = json.loads((OUT / 'latest_audit.json').read_text(encoding='utf-8'))
    original_path = Path(pointer['directory']) / 'results.json'
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != pointer['results_sha256']:
        raise ValueError('十年现金重建核验文件变化')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    baseline = original['ten_year_cash_restart']
    if baseline['strategy']['start'] != protocol['start'] or baseline['strategy']['end'] != protocol['end']:
        raise ValueError('基准回放区间不匹配')
    directory = OUT / 'action_sensitivity' / (pd.Timestamp.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {'protocol': protocol, 'candidate': selected['candidate'], 'policy': selected['policy'],
              'input_hashes': {**selected['hashes'], protocol_path.relative_to(BASE).as_posix(): hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                               original_path.relative_to(BASE).as_posix(): hashlib.sha256(original_path.read_bytes()).hexdigest(),
                               Path(__file__).relative_to(BASE).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    write_json_atomic(directory / 'frozen_protocol.json', frozen)
    tested = copy.copy(market)
    tested.bars, adjustments = adjusted_split_bars(market.bars, protocol['symbol'], protocol['events'])
    report, trades, issues = execute(tested, selected['candidate'], selected['policy'], protocol['start'])
    _, benchmark = relative_metrics(report, BASE / selected['benchmark'], selected['comparison'])
    result = {'baseline': baseline, 'tested': comparison(report.account, benchmark.spy_total_cny),
              'adjustments': adjustments, 'issues': issues,
              'ledger': ledger_audit(tested, report, trades, selected['policy'], selected['policy']['initial_capital']),
              'original_data_modified': False, 'corporate_actions_audited': False}
    fields = ('cagr', 'mean_daily_drawdown', 'mean_annual_max_drawdown', 'max_drawdown', 'cumulative_return')
    result['changes'] = {key: result['tested']['strategy'][key] - baseline['strategy'][key] for key in fields}
    report.to_csv(directory / 'report.csv')
    trades.to_csv(directory / 'trades.csv', index=False)
    write_json_atomic(directory / 'results.json', result)
    load_selected(SELECTED)
    write_json_atomic(OUT / 'action_sensitivity/latest.json', {'directory': str(directory)})
    print(json.dumps({'directory': str(directory), 'changes': result['changes'], 'checks': result['tested']['checks']}, indent=2))
    return directory


if __name__ == '__main__':
    run()
