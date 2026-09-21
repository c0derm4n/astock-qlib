"""对冻结现金账本做回撤时长与资产归因，不调参、不改策略、不重跑历史。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.goal30.audit import BASE, comparison
from research.goal30.verify_dividend_notices import digest, require

ARCHIVE = BASE / 'output/goal30/cash_replay/20260913-152542-31f263cf'
RISK = BASE / 'output/goal30/risk_overlay/20260913-215040-ce050fc8'
OUT = BASE / 'output/goal30/diagnostics/drawdown_duration'


def episodes(wealth, pnl=None):
    """按连续低于历史高点的交易日分段，未恢复段明确标记；归因用经济损益。"""
    require(wealth.index.is_monotonic_increasing and wealth.index.is_unique, '净值日期乱序或重复')
    require(np.isfinite(wealth).all() and wealth.gt(0).all(), '净值必须完整且为正')
    if pnl is not None:
        require(wealth.index.equals(pnl.index) and np.isfinite(pnl).all().all(), '损益日期或数值不符')
    high = wealth.cummax()
    dd = 1 - wealth / high
    rows, active, peak = [], None, 0

    def finish(end, recovery):
        block = dd.iloc[active:end + 1]
        trough = block.idxmax()
        row = {'peak_date': str(wealth.index[peak].date()),
               'underwater_start': str(wealth.index[active].date()),
               'underwater_end': str(wealth.index[end].date()),
               'recovery_date': str(wealth.index[recovery].date()) if recovery is not None else None,
               'recovered': recovery is not None, 'underwater_sessions': len(block),
               'max_drawdown': float(block.max()), 'trough_date': str(trough.date()),
               'contribution_to_full_mean_pp': float(block.sum() / len(wealth) * 100)}
        if pnl is not None:
            cumulative = pnl.iloc[active:end + 1].cumsum()
            base = wealth.iloc[peak]
            # 期间高水位固定，累计分资产经济损益除以峰值净值可精确分解回撤。
            error = (-cumulative.sum(axis=1) / base - block).abs().max()
            require(error < 1e-10, '回撤归因不能重建逐日损失幅度')
            row['asset_contribution_to_full_mean_pp'] = (-cumulative.sum() / base / len(wealth) * 100).to_dict()
            row['asset_loss_to_trough_pp'] = (-cumulative.loc[trough] / base * 100).to_dict()
        rows.append(row)

    for i, value in enumerate(dd):
        if value > 1e-12:
            if active is None:
                active = i
        else:
            if active is not None:
                finish(i - 1, i)
                active = None
            peak = i
    if active is not None:
        finish(len(wealth) - 1, None)
    require(abs(sum(r['contribution_to_full_mean_pp'] for r in rows) - dd.mean() * 100) < 1e-10,
            '回撤段贡献不能重建全期均值')
    return {'mean_daily_drawdown': float(dd.mean()), 'max_drawdown': float(dd.max()),
            'underwater_sessions': int(dd.gt(1e-12).sum()), 'observations': len(wealth),
            'episode_count': len(rows), 'completed_episode_count': sum(r['recovered'] for r in rows),
            'mean_episode_max_drawdown_including_open': float(np.mean([r['max_drawdown'] for r in rows])) if rows else 0.,
            'episodes': sorted(rows, key=lambda r: r['contribution_to_full_mean_pp'], reverse=True)}


def run():
    require(not OUT.exists(), '诊断目录已存在；读取既有结果，不能覆盖')
    inputs = {}

    def checked(path, expected=None):
        sha = digest(path)
        require(expected is None or sha == expected, f'输入哈希不符：{path}')
        inputs[path.relative_to(BASE).as_posix()] = sha
        return path

    frozen = json.loads(checked(ARCHIVE / 'frozen_protocol.json').read_text(encoding='utf-8'))
    for name, sha in frozen['input_hashes'].items():
        checked(BASE / name, sha)
    risk_results = json.loads(checked(RISK / 'results.json').read_text(encoding='utf-8'))
    completion = json.loads(checked(RISK / 'completion.json').read_text(encoding='utf-8'))
    require(digest(RISK / 'results.json') == completion['results_sha256'], '风险归档结果改变')
    case = next(iter(risk_results['cases'].values()))
    benchmark_name = next(name for name in case['artifact_hashes'] if name.startswith('benchmark_'))
    benchmark = pd.read_csv(checked(RISK / benchmark_name, case['artifact_hashes'][benchmark_name]),
                            index_col=0, parse_dates=True)['spy_total_cny']
    report = pd.read_csv(checked(ARCHIVE / 'report_cash_base.csv'), index_col=0, parse_dates=True)
    holdings = pd.read_csv(checked(ARCHIVE / 'holdings_cash_base.csv'), parse_dates=['date'])
    trades = pd.read_csv(checked(ARCHIVE / 'trades_cash_base.csv'), parse_dates=['date'])
    actions = pd.read_csv(checked(ARCHIVE / 'actions_cash_base.csv'), parse_dates=['date'])
    # 所有基础账本文件与已冻结风险登记的输入快照交叉核对。
    registration = json.loads(checked(RISK / 'registration.json').read_text(encoding='utf-8'))
    for name, sha in inputs.copy().items():
        if name in registration['input_hashes']:
            require(sha == registration['input_hashes'][name], f'基础账本改变：{name}')
    values = holdings.pivot(index='date', columns='symbol', values='value').reindex(report.index).fillna(0.)
    require(set(values.columns) == {'SH510300', 'SH515880', 'SH518880'}, '基线实际持仓标的改变')
    trades['cash_outflow'] = (trades.quantity * trades.price * trades.side.map({'buy': 1, 'sell': -1})
                              + trades.commission)
    flows = trades.groupby(['date', 'symbol']).cash_outflow.sum().unstack().reindex_like(values).fillna(0.)
    accrued = actions[actions.type.eq('accrue_dividend')].groupby(['date', 'symbol']).amount.sum()
    accrued = accrued.unstack().reindex_like(values).fillna(0.)
    pnl = values - values.shift(1, fill_value=0.) - flows + accrued
    expected_pnl = report.account.diff()
    expected_pnl.iloc[0] = report.account.iloc[0] - frozen['policy']['initial_capital']
    pnl_error = float((pnl.sum(axis=1) - expected_pnl).abs().max())
    require(pnl_error < 1e-6, '分资产经济损益与含应收净值不符')
    strategy, spy = episodes(report.account, pnl), episodes(benchmark)
    metrics = comparison(report.account, benchmark)
    asset_mean = {symbol: float(sum(r['asset_contribution_to_full_mean_pp'][symbol]
                                   for r in strategy['episodes'])) for symbol in pnl.columns}
    require(abs(sum(asset_mean.values()) - strategy['mean_daily_drawdown'] * 100) < 1e-10, '归因总额不符')
    checked(Path(__file__).resolve())
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'comparison': metrics,
              'strategy': strategy, 'spy_total_cny': spy, 'asset_contribution_to_mean_daily_drawdown_pp': asset_mean,
              'max_daily_pnl_reconstruction_error_cny': pnl_error,
              'method': '收盘市值变动减净买卖现金流和佣金，加除息应收权益；支付到账仅为内部转账，不再次确认收益。',
              'interpretation': '正值增加平均回撤、负值抵消；资产归因来自既有持仓路径，不是删除或替换该资产后的策略反事实。',
              'full_goal_achieved': False, 'investment_admission': False, 'input_hashes': inputs}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pnl.to_csv(OUT / 'asset_daily_economic_pnl.csv')
    lines = ['# 冻结候选的回撤时长与资产归因', '',
             '严格十年2016-09-12—2026-09-11；相同中国交易日、人民币口径，策略扣费、SPY含分红。', '',
             '| 对象 | 日均回撤 | 最大回撤 | 水下交易日 | 连续回撤段（含未恢复） |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, row in [('策略', strategy), ('人民币SPY', spy)]:
        lines.append(f"| {name} | {row['mean_daily_drawdown']:.2%} | {row['max_drawdown']:.2%} | "
                     f"{row['underwater_sessions']}/{row['observations']} | {row['episode_count']} |")
    lines += ['', '## 对策略全期日均回撤贡献最大的5段', '',
              '| 高点日期 | 谷底日期 | 恢复日期 | 水下交易日 | 段内最大回撤 | 对全期均值贡献 |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for row in strategy['episodes'][:5]:
        lines.append(f"| {row['peak_date']} | {row['trough_date']} | {row['recovery_date'] or '期末未恢复'} | "
                     f"{row['underwater_sessions']} | {row['max_drawdown']:.2%} | {row['contribution_to_full_mean_pp']:.2f}个百分点 |")
    lines += ['', '## 全期日均回撤的资产贡献', '', '| ETF | 贡献 |', '| --- | ---: |']
    for symbol, value in asset_mean.items():
        lines.append(f'| {symbol} | {value:+.2f}个百分点 |')
    lines += ['', result['method'], '', result['interpretation'], '',
              f'逐日经济损益最大重建误差{pnl_error:.10f}元；回撤段和资产贡献分别能重建日均回撤。', '',
              '每个连续回撤段的最大深度平均值另存JSON，并明确包含未恢复段；它与逐日平均回撤不同，不据此切换准入口径。', '',
              '**整体目标仍未完成。** 本诊断只定位历史风险来源，没有新增可采用策略或独立样本外证据。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'strategy_largest_episodes': strategy['episodes'][:3],
                      'asset_contribution_pp': asset_mean, 'pnl_error': pnl_error}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
