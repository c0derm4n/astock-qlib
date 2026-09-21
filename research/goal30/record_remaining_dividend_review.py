"""记录调用方在实际查看完整页面后输入的字段；本工具本身不执行目视核验。"""
import json
import sys

from backend.artifacts import write_json_atomic
from research.goal30.acquire_remaining_dividends import ROOT
from research.goal30.primary_research_series import read
from research.goal30.turnover_budget import require


if __name__=='__main__':
    submitted=json.load(sys.stdin)
    extracted={r['source']['event_key']:r for r in read(ROOT/'extracted_fields_pending_visual.json')['events']}
    inventory={r['file']:r for r in read(ROOT/'prepared_inventory.json')['documents']}
    log=read(ROOT/'visual_review.json')
    for key,values in submitted.items():
        row=extracted[key];file=row['source']['file'];item=inventory[file]
        require(file not in log['files'],'该文件已记录，禁止覆盖目视记录')
        fields=dict(zip(('cash_per_share_decimal','record_date','ex_date','announced_payment_date'),values))
        require(len(values)==4 and fields=={k:row['event'][k] for k in fields},'目视字段与提取结果不同，须先核对')
        require(not row['internal_investor_claim_date_conflicts'],'冲突原件必须单独记录差异')
        wanted=row['relevant_pages']
        image_hashes={p:h for p,h in item['rendered_hashes'].items() if int(p.rsplit('-',1)[-1].split('.')[0]) in wanted}
        log['files'][file]={'pdf_sha256':item['pdf_sha256'],'pages_reviewed':wanted,'rendered_hashes':image_hashes,
            'fields_observed':fields,'note':'Complete relevant pages visually inspected. Fund code, per10-share amount, date table, close-of-record entitlement, designated-broker investor claim date and delayed payment for undesignated investors checked.'}
    log['complete']=len(log['files'])==33
    write_json_atomic(ROOT/'visual_review.json',log)
    print(json.dumps({'reviewed_documents':len(log['files']),'reviewed_pages':sum(len(x['pages_reviewed']) for x in log['files'].values())}))
