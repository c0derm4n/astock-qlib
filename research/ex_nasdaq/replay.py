"""核对冻结规则、输入和验收产物后，独立重放30只池入选策略。"""
from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from pathlib import Path
import pandas as pd
from research.ex_nasdaq.core import market30, execute, SYMBOLS
from research.ex_nasdaq.explore import BASE, OUT, write_json
from research.sp500.validate import relative_metrics, ledger_audit


def load_selected(path):
    selected = json.loads(Path(path).read_text(encoding='utf-8'))
    directory = BASE/selected['validation']
    frozen = json.loads((directory/'frozen_validation.json').read_text(encoding='utf-8'))
    expected = {'candidate': frozen['candidate'], 'policy': frozen['policy'],
                'source': frozen['registry']['source'], 'benchmark': frozen['registry']['benchmark'],
                'starts': frozen['registry']['starts'], 'end': frozen['registry']['end'],
                'comparison': frozen['registry']['comparison'], 'symbols': sorted(SYMBOLS), 'hashes': frozen['hashes']}
    if any(selected.get(k) != v for k,v in expected.items()):
        raise ValueError('策略配置与冻结规则不一致，请登记新的实验')
    for relative, expected_hash in frozen['hashes'].items():
        if hashlib.sha256((BASE/relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'冻结输入已变化：{relative}')
    for name,h in json.loads((directory/'artifact_hashes.json').read_text(encoding='utf-8')).items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != h:
            raise ValueError(f'验证产物已变化：{name}')
    return selected, directory


def replay(path=None):
    selected, validation = load_selected(path or Path(__file__).with_name('selected_strategy.json'))
    market = market30(BASE/selected['source'])
    directory = OUT/'replays'/(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    result = {}
    for label, start in selected['starts'].items():
        frame, trades, issues = execute(market, selected['candidate'], selected['policy'], start)
        metrics, _ = relative_metrics(frame, BASE/selected['benchmark'], selected['comparison'])
        metrics['ledger'] = ledger_audit(market, frame, trades, selected['policy'], selected['policy']['initial_capital'])
        original = pd.read_csv(validation/f'report_base_{label}.csv', index_col=0, parse_dates=True)
        if not frame.index.equals(original.index):
            raise ValueError('重放日期不一致')
        metrics['reproduction_error_cny'] = float((frame.account-original.account).abs().max())
        if metrics['reproduction_error_cny'] > .001:
            raise ValueError('独立重放净值不一致')
        metrics['issues'] = issues
        frame.to_csv(directory/f'report_{label}.csv')
        trades.to_csv(directory/f'trades_{label}.csv', index=False)
        result[label] = metrics
        print(f"{label}: 年化{metrics['cagr']:.2%} 重放差额{metrics['reproduction_error_cny']:.8f}元", flush=True)
    write_json(directory/'results.json', result)
    write_json(OUT/'replays/latest.json', {'directory': str(directory)})
    print(f'独立重放完成：{directory}', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy')
    args = parser.parse_args()
    replay(args.strategy)
