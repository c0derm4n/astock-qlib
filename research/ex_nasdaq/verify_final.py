"""最终验收：独立算术核对、规则防篡改、全池隔离与时期依赖披露。"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
from research.ex_nasdaq.explore import BASE, OUT, write_json
from research.ex_nasdaq.core import market30, SYMBOLS
from research.ex_nasdaq.replay import load_selected


def verify():
    selected, validation = load_selected(Path(__file__).with_name('selected_strategy.json'))
    frozen = json.loads((validation/'frozen_validation.json').read_text(encoding='utf-8'))
    result = json.loads((validation/'results.json').read_text(encoding='utf-8'))
    checks = result['checks']
    assert checks['historical_acceptance']
    assert selected['starts'] == {'long': '2016-01-05', 'recent': '2021-01-04'}
    assert selected['end'] == '2026-09-11'
    assert selected['comparison'] == {'max_cagr_shortfall': .02, 'min_cumulative_ratio': .85}
    assert len(result['stress']) == 6 and len(result['neighbors']) == 5
    assert checks['neighbors_passing'] >= 4
    assert frozen['registry']['research_drawdown_budget'] == .3
    assert {j['name']: j['stress'] for j in frozen['jobs'] if j['section'] == 'stress'} == {
        'slippage_2': {'slippage_multiplier': 2}, 'slippage_3': {'slippage_multiplier': 3},
        'capital_10': {'capital_multiplier': 10}, 'capital_100': {'capital_multiplier': 100},
        'partial_50': {'fill_fraction': .5}, 'delay_2': {'delay': 2}}
    market = market30(BASE/selected['source'])
    assert len(SYMBOLS) == 30 and 'SH513100' not in SYMBOLS
    universe_audit = {'symbols': sorted(SYMBOLS), 'bars': len(market.bars), 'eligibility_rows': len(market.eligibility)}
    for name in ('bars','eligibility','nav','meta'):
        assert set(getattr(market,name).symbol) == SYMBOLS
    for name in ('features','labels','label_ends'):
        assert set(getattr(market,name).index.get_level_values('instrument')) == SYMBOLS
    market.eligibility.to_parquet(OUT/'verification/eligibility_30.parquet',index=False)
    write_json(OUT/'verification/universe_audit.json',universe_audit)
    replay = Path(json.loads((OUT/'replays/latest.json').read_text(encoding='utf-8'))['directory'])
    spy = pd.read_csv(BASE/selected['benchmark']/'spy.csv',index_col='date',parse_dates=True)
    fx = pd.read_csv(BASE/selected['benchmark']/'usdcny.csv',index_col='date',parse_dates=True)
    arithmetic = {}
    for label,start in selected['starts'].items():
        a = pd.read_csv(validation/f'report_base_{label}.csv',index_col=0,parse_dates=True)
        b = pd.read_csv(replay/f'report_{label}.csv',index_col=0,parse_dates=True)
        pd.testing.assert_frame_equal(a,b)
        years = (a.index[-1]-a.index[0]).days/365.25
        total = a.account.iloc[-1]/selected['policy']['initial_capital']-1
        annual = (1+total)**(1/years)-1
        bench_total = (spy.at[a.index[-1],'adjusted_close']/spy.at[a.index[0],'adjusted_close'] *
                       fx.at[a.index[-1],'close']/fx.at[a.index[0],'close'])-1
        bench_annual = (1+bench_total)**(1/years)-1
        assert abs(total-result['base'][f'base_{label}']['cumulative_return']) < 1e-10
        assert abs(annual-result['base'][f'base_{label}']['cagr']) < 1e-10
        assert annual-bench_annual >= -.02 and total/bench_total >= .85
        arithmetic[label] = {'strategy_total': total, 'strategy_cagr': annual, 'spy_cny_total': bench_total,
                             'spy_cny_cagr': bench_annual, 'independent_replay_difference_cny': float((a.account-b.account).abs().max())}
    rounds, trade_rows, trades_symbols = [],0,set()
    for directory in sorted((OUT/'runs').iterdir()):
        if not directory.is_dir():
            continue
        assert (directory/'completion.json').exists(), f'仍有未完成批次：{directory}'
        registry = json.loads((directory/'registry.json').read_text(encoding='utf-8'))
        data = json.loads((directory/'results.json').read_text(encoding='utf-8'))
        assert set(data) == {c['name'] for c in registry['candidates']}
        for file in directory.glob('trades_*.csv'):
            t = pd.read_csv(file)
            assert set(t.symbol).issubset(SYMBOLS)
            trade_rows += len(t); trades_symbols.update(t.symbol)
        rounds.append({'directory': directory.name,'candidates': len(data)})
    assert len(rounds) == 8 and sum(x['candidates'] for x in rounds) == 95
    for section in ('base','stress','neighbors','additional'):
        for name,metric in result[section].items():
            ledger = metric['ledger']
            assert ledger['max_nav_error_cny'] < .01 and ledger['max_cash_error_cny'] < .01
            assert not any(ledger[k] for k in ('timing_violations','ineligible_buys','nonlot_buys'))
            assert ledger['max_adv_participation'] <= .02 and ledger['minimum_cash'] >= -.01
            assert set(pd.read_csv(validation/f'trades_{name}.csv').symbol).issubset(SYMBOLS)
    bad = copy.deepcopy(selected)
    bad['candidate']['weights']['SH515880'] = .61
    bad_path = OUT/'verification/rejected_changed_strategy.json'
    write_json(bad_path,bad)
    process = subprocess.run([sys.executable,'-m','research.ex_nasdaq.replay','--strategy',str(bad_path)],
                             cwd=BASE,env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8')
    assert process.returncode != 0 and '策略配置与冻结规则不一致' in process.stderr
    write_json(OUT/'verification/config_rejection.json',{'returncode': process.returncode,'expected_error_observed':True})
    test_log = (OUT/'verification/unittest.txt').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 92 tests',test_log) and test_log.strip().endswith('OK')
    pointer = json.loads((BASE/'output/current_run.json').read_text(encoding='utf-8'))
    assert pointer['run_id'] == '20260912-220345-f20ea3f5'
    production = json.loads((BASE/selected['source']/'run_meta.json').read_text(encoding='utf-8'))
    for relative,h in production['code']['files'].items():
        assert hashlib.sha256((BASE/relative).read_bytes()).hexdigest() == h, relative
    a = pd.read_csv(validation/'report_base_long.csv',index_col=0,parse_dates=True)
    b = pd.read_csv(validation/'benchmark_base_long.csv',index_col=0,parse_dates=True)
    early = a.loc[:'2024-12-31']; years = (early.index[-1]-early.index[0]).days/365.25
    net = a['return']-a.cost
    dependence = {'through_2024_strategy_cagr': (early.account.iloc[-1]/1e6)**(1/years)-1,
                  'through_2024_spy_cny_cagr': b.loc[early.index].spy_total_cny.iloc[-1]**(1/years)-1,
                  '2025_strategy_return': (1+net[net.index.year==2025]).prod()-1,
                  'interpretation': '历史收益门槛通过高度依赖2025行情；不能证明稳定alpha或实盘准入。'}
    write_json(OUT/'verification/period_dependence.json',dependence)
    report_path = BASE/'docs/SP500_EX_NASDAQ_ACCEPTANCE.md'
    doc = report_path.read_text(encoding='utf-8')
    note = (f"**时期依赖限制：** 2025年策略收益为{dependence['2025_strategy_return']:.2%}；截至2024年底，2016年以来年化仅{dependence['through_2024_strategy_cagr']:.2%}，同期人民币SPY为{dependence['through_2024_spy_cny_cagr']:.2%}。最终对标达标高度依赖这段行情，不能据此认定可持续超额收益或实盘准入。数值核对见`output/sp500_ex_nasdaq/verification/period_dependence.json`。\n\n")
    if '**时期依赖限制：**' not in doc:
        doc = doc.replace('## 对标结果',note+'## 对标结果',1)
        report_path.write_text(doc,encoding='utf-8')
    for name in ('docs/SP500_EX_NASDAQ_ACCEPTANCE.md','docs/SP500_EX_NASDAQ_PLAN.md','research/ex_nasdaq/README.md'):
        file = BASE/name
        for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if not link.startswith(('http','#')):
                assert (file.parent/link.split('#')[0]).resolve().exists(), link
    chart = pd.read_csv(validation/'comparison_wealth.csv',index_col=0,parse_dates=True)
    assert (chart.iloc[:,0]-a.account/1e6).abs().max() < 1e-10
    image = BASE/'docs/images/sp500_ex_nasdaq_comparison.png'
    assert image.exists() and image.stat().st_size > 10000
    artifacts = [report_path,image,Path(__file__).with_name('selected_strategy.json'),validation/'results.json',
                 replay/'results.json',OUT/'verification/period_dependence.json',OUT/'verification/unittest.txt',Path(__file__)]
    completion = {'historical_goal_verified': True,'investment_admission':False,'required_input_hashes_verified':len(frozen['hashes']),
                  'validation_artifact_hashes_verified':len(json.loads((validation/'artifact_hashes.json').read_text(encoding='utf-8'))),
                  'validation':str(validation),'replay':str(replay),'independent_arithmetic':arithmetic,'checks':checks,
                  'tests_passed':92,'config_tampering_rejected':True,'rounds':rounds,'trades_checked_across_exploration':trade_rows,
                  'all_exploration_traded_symbols':sorted(trades_symbols),'excluded_symbol_trades':0,'production_pointer_unchanged':True,
                  'period_dependence':dependence,'artifact_hashes':{p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    write_json(OUT/'completion_verification.json',completion)
    print(json.dumps({k:completion[k] for k in ['historical_goal_verified','investment_admission','required_input_hashes_verified','validation_artifact_hashes_verified','tests_passed','config_tampering_rejected','excluded_symbol_trades']},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    verify()
