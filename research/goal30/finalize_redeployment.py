"""验收资金重新分配阶段的证据链、报告与保留的投资门槛。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

from src.artifacts import write_json_atomic
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import read
from research.goal30.run_redeployment import OUT
from research.goal30.turnover_budget import require


def finalize(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(OUT), '需使用本研究登记目录')
    audit = read(directory/'outcome_audit.json'); proof = read(directory/'verification.json')
    tests = read(directory/'tests_verification.json'); manifest = read(directory/'report_manifest.json')
    require(audit['passed'] and proof['passed'] and tests['passed'], '仍有未通过核验')
    require(tests['test_groups_run'] == 6 and not tests['errors'] and not tests['failures'], '分配测试未全部通过')
    require(proof['cases_verified'] == 15 and proof['ledger_rows'] == 36420 and proof['signals_verified'] == 36405,
            '预登记情景或逐日核对不完整')
    require(audit['paired_report_rows'] == 72840 and audit['same_sessions'] == 2428, '新旧配对核验不完整')
    require(manifest['figure_visually_reviewed'] and manifest.get('figure_reviewed_sha256') ==
            digest(directory/'paired_comparison.png'), '比较图未目视核验或图已改变')
    hashes = {**audit['input_and_artifact_hashes'], **manifest['source_hashes'], **manifest['report_hashes'], **tests['source_hashes']}
    for rel, expected in hashes.items():
        require(digest(BASE/rel) == expected, '最终证据链变化：'+rel)
    pointer = read(BASE/'output/current_run.json')
    require(pointer['run_id'] == '20260912-220345-f20ea3f5', '正式策略指针变化')
    require(not audit['primary_and_neighbors_gate']['eligible_for_further_admission_review'] and
            audit['full_history_joint_performance_passes'] == 0, '此验收报告仅适用于本批失败结果')
    plan = BASE/'docs/SP500_REDEPLOYMENT_PLAN.md'
    plan_text = plan.read_text(encoding='utf-8')
    require(plan_text.count('[x]') == 5 and '[ ]' not in plan_text, '本阶段研究计划尚未完成')
    doc = BASE/'docs/SP500_REDEPLOYMENT_ACCEPTANCE.md'
    links = []
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
        if target.startswith(('https://', 'http://', '#')):
            continue
        path = (doc.parent/unquote(target.strip('<>').split('#', 1)[0])).resolve()
        require(path.is_relative_to(BASE) and path.is_file(), '报告本地链接无效：'+target)
        links.append(path.relative_to(BASE).as_posix())
    files = [directory/n for n in ('outcome_audit.json', 'verification.json', 'report_manifest.json', 'tests_verification.json')]
    files.extend([plan, Path(__file__), BASE/'output/current_run.json'])
    hashes.update({p.relative_to(BASE).as_posix(): digest(p) for p in files})
    status = {'stage_completed': True, 'accepted_at': __import__('datetime').datetime.now().astimezone().isoformat(),
        'registered_cases': 15, 'test_groups_passed': 6, 'ledger_rows_verified': 36420,
        'signals_verified': 36405, 'paired_report_rows_verified': 72840,
        'benchmark_rebuilt_from_raw_spy_fx': True, 'figure_visually_reviewed': True,
        'report_local_links_verified': links, 'full_history_joint_performance_passes': 0,
        'post_budget_target_risk_exceedance_case_days': proof['post_budget_target_risk_exceedance_case_days'],
        'primary_and_neighbors_gate': audit['primary_and_neighbors_gate'],
        'formal_run_id': pointer['run_id'], 'daily_candidate_changed': False,
        'qualified_real_forward_sessions': 0, 'full_actions_and_survival_directory_audited': False,
        'actual_broker_receipts_confirmed': False, 'investment_admission': False, 'full_goal_achieved': False,
        'next_work': 'Verify the eight known post-listing missing-quote symbol-days using source notices or actual market data; distinguish confirmed nontrading from unresolved missing quotes before any signal-series change. Preserve current research failures and do not infer performance improvement.',
        'input_and_artifact_hashes': hashes}
    dest = BASE/'output/goal30/verification/redeployment'; dest.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dest/'requirements_audit.json', status)
    print(json.dumps({k:v for k,v in status.items() if k != 'input_and_artifact_hashes'}, ensure_ascii=False), flush=True)
    print(json.dumps({'verified_hashes': len(hashes)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    finalize(parser.parse_args().directory)
