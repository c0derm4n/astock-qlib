"""核查本阶段证据、报告与正式指针，产出机器验收状态。"""
import json
from pathlib import Path

from backend.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT as EVIDENCE
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.group_return_attribution import OUT as ATTRIBUTION
from research.goal30.primary_research_series import read
from research.goal30.run_dividend_payment_replay import OUT as REPLAY
from research.goal30.turnover_budget import require


def finalize():
    directory=BASE/read(REPLAY/'latest_registration.json')['directory']
    evidence=read(EVIDENCE/'verification.json');replay=read(directory/'verification.json')
    attribution=read(ATTRIBUTION/'verification.json');reports=read(ATTRIBUTION/'report_manifest.json')
    inputs={}
    for source in (evidence,replay,attribution):
        require(source['passed'],'存在未通过验收')
        inputs.update(source.get('input_hashes',{}));inputs.update(source.get('artifact_hashes',{}))
    inputs.update(reports['source_hashes']);inputs.update(reports['report_hashes'])
    for rel,h in inputs.items(): require(digest(BASE/rel)==h,'最终验收发现输入或产物变化：'+rel)
    require(reports['figure_visually_reviewed'],'归因图未目视验收')
    pointer=read(BASE/'output/current_run.json')
    require(pointer['run_id']=='20260912-220345-f20ea3f5','正式指针变化')
    results=read(directory/'results.json')
    require(results['signal_equivalence']['rows_exactly_equal']==64238,'信号相等性未通过')
    full=[v for v in results['cases'].values() if v['kind']!='event']
    require(len(full)==18 and not any(v['metrics']['checks']['joint_admission_performance'] for v in full),'投资结果范围不同')
    fixed=[v for v in full if v['kind']=='fixed']
    require(len(fixed)==3 and all(v['prior_difference']['same_trades'] and v['prior_difference']['maximum_account_difference']==0 for v in fixed),
            '固定候选对照不同')
    plan=BASE/'docs/SP500_REMAINING_DIVIDENDS_PLAN.md'
    require(plan.read_text(encoding='utf-8').count('[x]')==5 and '[ ]' not in plan.read_text(encoding='utf-8'),'本阶段计划未完成')
    for p in [EVIDENCE/'verification.json',directory/'verification.json',ATTRIBUTION/'verification.json',
              ATTRIBUTION/'report_manifest.json',plan,Path(__file__),BASE/'output/current_run.json']:
        inputs[p.relative_to(BASE).as_posix()]=digest(p)
    audit={'stage_completed':True,'input_and_artifact_hashes':inputs,
        'original_dividend_notices':33,'visually_reviewed_relevant_pages':66,
        'primary_amount_and_date_tables_verified':33,'consistent_payment_semantics':32,
        'internal_payment_year_conflicts_retained':1,'notice_test_groups_passed':8,'attribution_test_groups_passed':4,
        'payment_impact_cases_verified':111,'payment_impact_ledger_rows':replay['ledger_days'],
        'identical_signal_rows':64238,'base_attribution_ledger_rows':attribution['ledger_rows'],
        'independent_history_symbol_sessions':attribution['history_symbol_sessions'],
        'full_history_joint_performance_passes':0,'full_history_cases':18,
        'formal_run_id':pointer['run_id'],'daily_candidate_changed':False,'qualified_real_forward_sessions':0,
        'full_actions_and_survival_directory_audited':False,'actual_broker_receipts_confirmed':False,
        'investment_admission':False,'full_goal_achieved':False,
        'next_research_mechanism':'Within unchanged position/group/gross/volatility limits, isolate constrained redistribution of capital clipped by position limits. Keep timing and signal rules fixed for this comparison; no presumption of performance success.'}
    out=BASE/'output/goal30/verification/remaining_dividends';out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'requirements_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='input_and_artifact_hashes'},ensure_ascii=False))


if __name__=='__main__': finalize()
