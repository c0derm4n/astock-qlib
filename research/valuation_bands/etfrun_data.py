"""从ETF.run公开页面归档提取日频等权估值；不冒充整体法或MRQ口径。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_snapshot, sha256_file, build_calendar
from .fetch import _next_session_19

FIELDS = {'000300': ('equalWeightedPeTtm', 'PE_TTM_EQUAL'),
          '399006': ('equalWeightedPbTtm', 'PB_ETFRUN_EQUAL')}
URLS = {'000300': 'https://www.etf.run/index/SSE/000300',
        '399006': 'https://www.etf.run/index/SZSE/399006'}


def extract_daily(html: str) -> pd.DataFrame:
    """仅解析公开HTML内已经展示的历史，拒绝把预计算全历史分位用于信号。"""
    def visit(obj):
        if isinstance(obj, dict):
            if 'indexDailyItems' in obj:
                return obj['indexDailyItems']
            children = obj.values()
        elif isinstance(obj, list):
            children = obj
        else:
            return None
        for child in children:
            found = visit(child)
            if found is not None:
                return found
        return None
    for match in re.finditer(r'self\.__next_f\.push\((\[.*?\])\)</script>', html, re.S):
        packet = json.loads(match.group(1))
        if len(packet) < 2 or not isinstance(packet[1], str) or 'indexDailyItems' not in packet[1]:
            continue
        block = packet[1]
        found = visit(json.loads(block[block.index(':') + 1:]))
        if found is not None:
            frame = pd.DataFrame(found)
            frame['date'] = pd.to_datetime(frame['date'])
            if frame.date.duplicated().any():
                raise ValueError('公开日频历史日期重复')
            return frame.sort_values('date')
    raise ValueError('未找到公开indexDailyItems，拒绝编造历史')


def build_snapshot(parent_manifest: Path, raw_files: dict[str, Path], destination: Path) -> Path:
    parent = load_snapshot(parent_manifest)
    destination.mkdir(parents=True, exist_ok=False)
    calendar = build_calendar('2010-01-01', '2026-11-06')
    prices, values, sources = [], [], []
    for code, raw in raw_files.items():
        field, metric = FIELDS[code]
        frame = extract_daily(raw.read_text(encoding='utf-8'))
        frame = frame[frame.date.between('2011-01-01', '2026-11-06')].copy()
        expected = calendar[calendar >= pd.Timestamp('2011-01-01')]
        if not pd.DatetimeIndex(frame.date).equals(expected):
            raise ValueError(f'{code} 日频日期不完整')
        if not np.isfinite(frame[[field, 'close']]).all().all() or (frame[[field, 'close']] <= 0).any().any():
            raise ValueError(f'{code} 估值或价格有无效值')
        stamp = _next_session_19(calendar, frame.date)
        digest = sha256_file(raw)
        prices.append(pd.DataFrame({'index_id': code, 'trade_date': frame.date, 'close': frame.close,
                                    'price_type': 'price_index_close', 'source_id': 'etfrun_public_html',
                                    'effective_available_at': stamp, 'availability_basis': 'assumed_next_session_19',
                                    'raw_sha256': digest}))
        values.append(pd.DataFrame({
            'index_id': code, 'trade_date': frame.date, 'metric': metric, 'value': frame[field],
            'aggregation_method': 'provider_declared_equal_weight',
            'financial_basis': 'TTM' if code == '000300' else 'provider_equalWeightedPbTtm_MRQ_unverified',
            'source_id': 'etfrun_public_html', 'source_published_at': '',
            'effective_available_at': stamp, 'availability_basis': 'assumed_next_session_19',
            'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
            'revision_id': 'downloaded_20261109_history_reconstructed', 'raw_sha256': digest}))
        sources.append({'series': f'index_valuation:{code}:{metric}', 'url': URLS[code],
                        'raw_file': str(raw.resolve()), 'raw_sha256': digest, 'raw_field': field,
                        'rows': len(frame), 'evidence_grade': 'historical_reconstructed', 'pit_verified': False,
                        'note': '公开页面声明等权；未公开完整财报/历史成分算法；不改名整体法或PB_MRQ'})
    frames = dict(parent.frames)
    frames['index_price'] = pd.concat(prices, ignore_index=True)
    frames['index_valuation'] = pd.concat(values, ignore_index=True)
    manifest = dict(parent.manifest)
    manifest.update({'data_version': destination.name, 'dataset_kind': 'real',
                     'created_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
                     'parent_manifest_path': str(parent_manifest.resolve()),
                     'parent_manifest_sha256': sha256_file(parent_manifest),
                     'sources': sources + [s for s in parent.manifest.get('sources', [])
                                            if str(s.get('series', '')).startswith(('etf_',))],
                     'valuation_version': 'etfrun_equal_weight_user_authorized_20261109',
                     'calendar': {'exchange': 'XSHG', 'start': '2010-01-01', 'end': '2026-11-06'},
                     'files': []})
    for name, frame in frames.items():
        if frame.empty:
            continue
        path = destination / f'{name}.csv'
        frame.to_csv(path, index=False, encoding='utf-8')
        manifest['files'].append({'name': name, 'path': path.name, 'sha256': sha256_file(path), 'rows': len(frame)})
    out = destination / 'manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--raw-a', type=Path, required=True)
    parser.add_argument('--raw-b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build_snapshot(args.parent_manifest, {'000300': args.raw_a, '399006': args.raw_b}, args.output))


if __name__ == '__main__':
    main()
