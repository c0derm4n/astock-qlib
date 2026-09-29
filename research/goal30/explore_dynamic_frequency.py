"""按用户修订目标复核旧动态方案，预登记并回放月/周/日动态重选。"""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.audit_redeployment_outcome import benchmark_from_raw
from research.goal30.bond_dividend_evidence import digest
from research.goal30.dynamic_frequency import decision, period
from research.goal30.group_diversification import decision as legacy
from research.goal30.primary_research_series import read, corrected_market
from research.goal30.report_current_candidate_yearly import annual_table
from research.goal30.run_group_diversification import feature_bank, check_inputs
from research.goal30.run_redeployment import drive, PRIOR
from research.goal30.validate_primary_execution import PrimaryMarketDays, inspect_case
from research.goal30.verify_turnover_replay import independent_metrics

ROOT = BASE/'output/goal30/dynamic_frequency'


def assess(data, benchmark):
    report = pd.DataFrame(data['report']).set_index('date'); report.index = pd.to_datetime(report.index)
    a = independent_metrics(report.account); b = independent_metrics(benchmark)
    years = annual_table(report.account, benchmark)
    failures = [r['year'] for r in years if r['strategy_max_drawdown'] > r['sp500_cny_total_max_drawdown']+1e-12]
    return {'strategy': a, 'benchmark': b, 'annual': years,
        'required_cagr': .5*b['cagr'], 'cagr_pass': a['cagr'] >= .5*b['cagr']-1e-12,
        'annual_drawdown_failed_years': failures,
        'user_historical_target_met': a['cagr'] >= .5*b['cagr']-1e-12 and not failures,
        'distinct_traded_etfs': len({r['symbol'] for r in data['trades']}), 'trades': len(data['trades'])}


def tests(prices, policy):
    assert period(pd.Timestamp('2024-12-30'),'weekly') == period(pd.Timestamp('2025-01-03'),'weekly')
    assert period(pd.Timestamp('2025-01-03'),'weekly') != period(pd.Timestamp('2025-01-06'),'weekly')
    scores = {'SH515880': 3., 'SH512800': 2., 'SH518880': 1.}
    synthetic = pd.DataFrame({s:100*np.exp(np.arange(280)*(.0001+i*.0001)) for i,s in enumerate(scores)},
                            index=pd.bdate_range('2023-01-02', periods=280))
    for frequency in ('monthly','weekly','daily'):
        v = {'frequency':frequency, 'grouped':True, 'slots':4, 'sizing':'risk_parity'}
        before = {}; state,target,trace = decision(before,scores,synthetic,set(scores),[],{},policy,v)
        assert before == {} and trace['selection_review']
        old = legacy({},scores,synthetic,set(scores),[],{},policy,v)
        assert target == old[1] and state['raw_weights'] == old[0]['raw_weights']
        repeat = decision(json.loads(json.dumps(state)),scores,synthetic,set(scores),[],{},policy,v)
        assert not repeat[2]['selection_review']
        _,exit_target,_ = decision(state,scores,synthetic,set(scores),['SH515880'],{},policy,v)
        assert 'SH515880' not in exit_target
    return {'passed':True,'checks':['week_spans_calendar_year','week_boundary','three_frequencies_first_selection_equal',
        'state_input_unchanged','json_restore_no_duplicate_review','exiting_target_removed']}


def run():
    benchmark, raw_files = benchmark_from_raw()
    oldreg = read(PRIOR/'registration.json')
    previous_stage = read(BASE/'output/goal30/verification/redeployment/requirements_audit.json')
    require = lambda condition,msg: None if condition else (_ for _ in ()).throw(ValueError(msg))
    require(previous_stage['stage_completed'], '上一阶段未完成')
    check_inputs({'input_hashes': previous_stage['input_and_artifact_hashes']})
    frames = PrimaryMarketDays(corrected_market())
    prices, features, bank, feature_proof = feature_bank(frames)
    test_proof = tests(prices, oldreg['policy'])
    variants = [{'name':f'{f}_group4_risk_parity','frequency':f,'grouped':True,'slots':4,'sizing':'risk_parity'}
                for f in ('monthly','weekly','daily')]
    directory = ROOT/pd.Timestamp.now().strftime('%Y%m%d-%H%M%S'); directory.mkdir(parents=True,exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name('dynamic_frequency.py'), BASE/'docs/SP500_DYNAMIC_V2_PLAN.md',
             *raw_files, PRIOR/'results.json', PRIOR/'registration.json']
    hashes = {**previous_stage['input_and_artifact_hashes'], **{p.relative_to(BASE).as_posix():digest(p) for p in files}}
    reg = {**oldreg, 'variants':variants, 'input_hashes':hashes, 'registered_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'primary':'weekly_group4_risk_parity', 'new_gate':'CAGR >= half CNY SPY; each common calendar-year slice MDD <= same-year SPY; include partial endpoints',
        'tests':test_proof,'feature_verification':feature_proof,'case_count':9}
    write_json_atomic(directory/'registration.json', reg)
    write_json_atomic(ROOT/'latest.json', {'directory':directory.relative_to(BASE).as_posix()})
    prior_assessed = {}
    for source in (PRIOR, BASE/'output/goal30/redeployment/20260914-090231-e8e11201'):
        for key,item in read(source/'results.json')['cases'].items():
            if item.get('kind') in ('event','fixed'): continue
            path = source/item['artifact']; require(digest(path)==item['sha256'],'旧情景变化')
            prior_assessed[source.parent.name+'/'+key] = assess(read(path), benchmark)
    write_json_atomic(directory/'prior_dynamic_reassessment.json', prior_assessed)
    print(json.dumps({'registered':str(directory.relative_to(BASE)), 'prior_dynamic_cases':len(prior_assessed),
        'prior_passes':sum(x['user_historical_target_met'] for x in prior_assessed.values()),'new_cases':9}),flush=True)
    results = {}
    for variant in variants:
        for scenario in reg['scenarios']:
            key = variant['name']+'__'+scenario['name']
            with patch('research.goal30.run_redeployment.decision', decision):
                data = drive(frames,prices,features,reg,variant,scenario)
            ledger = inspect_case(frames,reg,scenario,reg['actions'],data)
            # 用独立历史分数前缀重算信号，核对重选周期与目标。
            state = {}; reviews = 0
            for row in data['signals']:
                state,targets,trace = decision(state,bank[row['date']],prices.loc[:row['date']],row['eligible'],
                    row['exiting'],row['current_weights'],reg['policy'],variant)
                require(state==row['selection_state'] and targets.keys()==row['target_weights'].keys() and
                    all(abs(targets[s]-row['target_weights'][s])<1e-9 for s in targets),'独立前缀目标不同')
                reviews += trace['selection_review']
            if variant['frequency']=='monthly':
                old = read(PRIOR/('group4_risk_parity__'+scenario['name']+'.json'))
                require(data['report']==old['report'] and data['trades']==old['trades'],'月度对照未保持原结果')
            path = directory/(key+'.json'); write_json_atomic(path,data)
            saved = read(path)
            require(inspect_case(frames,reg,scenario,reg['actions'],saved)==ledger,'落盘独立账本不同')
            performance = assess(saved,benchmark)
            results[key] = {**performance,'ledger':ledger,'reviews':reviews,'signals_checked':len(data['signals']),
                'artifact':path.name,'sha256':digest(path),'variant':variant,'scenario':scenario}
            write_json_atomic(directory/'results.json',results)
            print(json.dumps({'case':key,'cagr':performance['strategy']['cagr'],
                'failed_years':performance['annual_drawdown_failed_years'],'passed':performance['user_historical_target_met'],
                'traded_etfs':performance['distinct_traded_etfs'],'reviews':reviews}),flush=True)
    check_inputs(reg)
    write_json_atomic(directory/'completion.json',{'cases':len(results),'registration_sha256':digest(directory/'registration.json'),
        'results_sha256':digest(directory/'results.json'),'passed_cases':sum(r['user_historical_target_met'] for r in results.values()),
        'investment_admission':False,'full_goal_achieved':False})


if __name__ == '__main__': run()
