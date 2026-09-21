"""从已验收剩余原件生成新行动版本；不覆盖旧版本或改变金额、除息。"""
import copy
from decimal import Decimal
from pathlib import Path

import pandas as pd

from research.goal30.acquire_remaining_dividends import ROOT as EVIDENCE
from research.goal30.audit import BASE
from research.goal30.bond_dividend_evidence import digest
from research.goal30.primary_research_series import ROOT, SOURCE, read, rebuild_series
from research.goal30.turnover_budget import require


def upgraded_actions():
    proof=read(EVIDENCE/'verification.json')
    require(proof['passed'] and proof['documents']==33 and proof['visual_pages']==66,'剩余原件尚未验收')
    for rel,h in {**proof['input_hashes'],**proof['artifact_hashes']}.items():
        require(digest(BASE/rel)==h,'已验收原件输入改变：'+rel)
    verified=read(EVIDENCE/'verified_remaining_dividends.json')['events']
    lookup={r['event']['symbol']+'_'+r['event']['ex_date']:r for r in verified}
    events=copy.deepcopy(read(ROOT/'actions.json')['events']);changes=[]
    for event in events:
        key=event['symbol']+'_'+event.get('ex_date','')
        if key not in lookup: continue
        item=lookup[key];new=item['event'];previous=copy.deepcopy(event)
        require(event['evidence_level']=='secondary' and Decimal(event['cash_per_share_decimal'])==Decimal(new['cash_per_share_decimal']),
                '升级项原级别或金额不同')
        if not event['signal_only']:
            require(all(event[k]==new[k] for k in ('record_date','ex_date','announced_payment_date','entitlement_basis')),'原件日期/权益不同')
        event.update(new)
        event.update(evidence_level='primary',evidence_file=(EVIDENCE/'verified_remaining_dividends.json').relative_to(BASE).as_posix(),
                     notice_file=(EVIDENCE/item['source']['file']).relative_to(BASE).as_posix(),
                     payment_semantics_verified=item['payment_semantics_verified'],
                     internal_investor_claim_date_conflicts=item['internal_investor_claim_date_conflicts'])
        event.pop('scope_note',None)
        changes.append({'event_id':event['event_id'],'signal_only':event['signal_only'],
                        'payment_semantics_before':previous.get('payment_semantics'),
                        'payment_semantics_after':event['payment_semantics'],
                        'remove_extra_model_delay':not event['signal_only'] and item['payment_semantics_verified']})
    require(len(changes)==33 and sum(c['remove_extra_model_delay'] for c in changes)==30,'升级范围改变')
    require(len(events)==69 and all(e['evidence_level']=='primary' for e in events),'行动数量或原件范围不同')
    return events,changes


def signal_equivalence(events):
    old=pd.read_parquet(ROOT/'signal_series.parquet')
    raw=pd.read_parquet(SOURCE/'market_snapshot.parquet')
    raw=raw[raw.symbol.isin(old.symbol.unique())]
    new,applied=rebuild_series(raw,events)
    pd.testing.assert_frame_equal(old,new,check_exact=True)
    require(len(applied)==69,'信号行动覆盖变化')
    return {'rows_exactly_equal':len(new),'symbols':int(new.symbol.nunique()),'events_applied':len(applied),
            'signal_amount_and_ex_dates_unchanged':True,'no_signal_series_file_overwritten':True}
