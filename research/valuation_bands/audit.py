"""复算正式归档净值并与真实研究结果比较；禁止把合成曲线用作绩效。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_calendar, sha256_file

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'A_pe_weekly': '新增沪深300PE', 'B_pb_weekly': '新增创业板PB',
          'lightgbm': '正式LightGBM核心卫星', 'trend200_vol15': '正式通信/黄金'}


def independent_metrics(frame, initial=1_000_000.0):
    """直接从日净值复算，不调用引擎或正式服务指标函数。"""
    frame = frame.sort_index()
    if frame.index.has_duplicates or frame.empty:
        raise ValueError('净值为空或日期重复')
    nav = frame.astype(float)
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError('净值非正或非有限')
    expected = build_calendar(nav.index[0], nav.index[-1])
    if not nav.index.equals(expected):
        raise ValueError('日净值缺少交易日，不能作完整账户比较')
    years = (nav.index[-1] - nav.index[0]).days / 365.2425
    drawdown = 1 - nav / np.maximum(initial, nav.cummax())
    annual = []
    for year, group in nav.groupby(nav.index.year):
        complete = (nav.index[0] <= pd.Timestamp(year, 1, 1) and
                    nav.index[-1] >= build_calendar(f'{year}-01-01', f'{year}-12-31')[-1])
        prior = nav[nav.index < pd.Timestamp(year, 1, 1)]
        base = float(prior.iloc[-1]) if len(prior) else initial
        annual.append({'year': int(year), 'return': float(group.iloc[-1] / base - 1),
                       'complete': bool(complete)})
    completed = [a for a in annual if a['complete']]
    worst = min(completed, key=lambda a: a['return']) if completed else None
    return {'start': str(nav.index[0].date()), 'end': str(nav.index[-1].date()),
            'observations': len(nav), 'initial_capital': initial,
            'final_nav': float(nav.iloc[-1]), 'cagr': float((nav.iloc[-1] / initial) ** (1 / years) - 1),
            'max_drawdown': float(drawdown.max()), 'worst_full_year': worst['return'] if worst else None,
            'worst_year': worst['year'] if worst else None, 'annual': annual}


def compare(research_run: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    pointer = ROOT / 'output/strategies/current.json'
    formal = ROOT / json.loads(pointer.read_text(encoding='utf-8'))['directory']
    reg = json.loads((research_run / 'registration.json').read_text(encoding='utf-8'))
    equal_weight = 'etfrun-equal' in str(reg.get('protocol_id', ''))
    labels = dict(LABELS)
    if equal_weight:
        labels.update({'A_pe_weekly': '新增沪深300PE（等权版）', 'B_pb_weekly': '新增创业板PB（等权版）'})
    rows, annual_rows, provenance = [], [], {str(pointer.relative_to(ROOT)): sha256_file(pointer)}
    formal_frames = {}
    for sid in ('lightgbm', 'trend200_vol15'):
        p = formal / f'{sid}_report.csv'
        f = pd.read_csv(p, parse_dates=['date']).set_index('date')['account']
        formal_frames[sid] = f
        provenance[str(p.relative_to(ROOT))] = sha256_file(p)
    start = max(f.index.min() for f in formal_frames.values())
    end = min(f.index.max() for f in formal_frames.values())
    new_curves = {}
    for sid in ('A_pe_weekly', 'B_pb_weekly'):
        case_dir = research_run / 'cases' / sid
        status = json.loads((case_dir / 'case_status.json').read_text(encoding='utf-8'))
        if (reg.get('dataset_kind') == 'real' and not reg.get('synthetic') and
                status['status'] in ('historical_run_complete', 'historical_goals_passed')):
            f = pd.read_csv(case_dir / 'daily_account.csv', parse_dates=['session']).set_index('session')['nav']
            new_curves[sid] = f
            start = max(start, f.index.min())
            end = min(end, f.index.max())
        else:
            rows.append({'strategy_id': sid, 'label': labels[sid], 'status': status['status'],
                         'reason': status.get('detail'), 'account_basis': '实际份额/现金账本（真实运行尚未完成）',
                         'cagr': None, 'max_drawdown': None, 'worst_full_year': None, 'final_nav': None})
    if start != pd.Timestamp('2016-09-12'):
        raise ValueError('共同起点改变须明确处理起始持仓，禁止静默重置资金')
    for sid, frame in {**formal_frames, **new_curves}.items():
        m = independent_metrics(frame.loc[start:end])
        if sid in formal_frames:
            archived = json.loads((formal / f'{sid}.json').read_text(encoding='utf-8'))
            basis = archived['account_basis']
            if end == frame.index[-1]:
                assert abs(m['final_nav'] - archived['metrics']['final_account']) < .01
                assert abs(m['max_drawdown'] - archived['metrics']['max_drawdown']) < 1e-8
                assert abs(m['worst_full_year'] - archived['metrics']['worst_complete_year']['return']) < 1e-8
        else:
            basis = '实际份额、分红应收及到账现金'
        annual_rows.extend({'strategy_id': sid, **a} for a in m.pop('annual'))
        rows.append({'strategy_id': sid, 'label': labels[sid], 'status': 'archived_nav_recomputed' if sid in formal_frames
                     else 'historical_run_complete', 'account_basis': basis, **m})
    rows.sort(key=lambda r: list(LABELS).index(r['strategy_id']))
    pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(annual_rows).to_csv(output / 'annual_returns.csv', index=False, encoding='utf-8-sig')
    payload = {'comparison_scope': '共同日期；账户执行口径存在差异；新增无真实数据时不作绩效排名',
               'start': str(start.date()), 'end': str(end.date()), 'formal_run': str(formal),
               'research_run': str(research_run), 'cagr_basis': 'actual_calendar_days / 365.2425',
               'valuation_version': 'etfrun_equal_weight' if equal_weight else 'original_protocol',
               'valuation_caveat': ('用户授权的等权版本，非原整体法；PB字段equalWeightedPbTtm未证实MRQ；历史重建非PIT'
                                     if equal_weight else None),
               'rows': rows, 'input_hashes': provenance, 'investment_admission': False}
    payload['comparison_code_sha256'] = sha256_file(Path(__file__))
    (output / 'comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--research-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.research_run.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
