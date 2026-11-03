"""Replay immutable 64-case exact posterior evidence; never edit old results."""
import json
from fractions import Fraction as F
from pathlib import Path
from stegatext_runtime.imec import IMECSession
from stegatext_runtime.experiments.engine import message_for
from tools.sentence_imec.analyze import replay

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/framework_contract_20261018/audit'

def main():
    rows=[]
    for job in json.loads((ROOT/'reports/sentence_imec_20261018/runs.json').read_text()):
        directory=ROOT/job['directory']
        cfg=json.loads((directory/'manifest.json').read_text())['configuration']
        for row in json.loads((directory/'results.json').read_text())['rows']:
            if row['variant']['kind']=='ordinary': continue
            s=json.loads((directory/'sender'/f"{row['case_id']}.json").read_text())['result']
            r=json.loads((directory/'receiver'/f"{row['case_id']}.json").read_text())['result']
            bits=message_for(cfg,row['seed'])
            verification=replay(s,bits,row['variant']['coder_parameters'])
            session=IMECSession(**row['variant']['coder_parameters'])
            state=s['coder_state']
            session.head=state['head_block']
            session.beliefs=[tuple(dict((i,F(v)) for i,v in p).get(i,F(0)) for i in range(1<<session.block_size)) for p in state['beliefs']]
            before=session.state()
            extra=session.finalize_confirmed_prefix()
            assert session.state()==before and extra==session.finalize_confirmed_prefix()
            assert r['coder_state']==state and r['bits']==s['bits']
            full=s['bits']+extra
            assert full==bits[:len(full)]
            rows.append(dict(experiment=job['config'],case_id=row['case_id'],online_committed_bits=len(s['bits']),
                total_confirmed_bits=len(full),prefix_exact=True,
                nonempty_payload=bool(full),**session.prefix_audit(),**verification))
    totals=dict(cases=len(rows),online=sum(r['online_committed_bits'] for r in rows),
                terminal=sum(r['terminal_extra_prefix_bits'] for r in rows),total=sum(r['total_confirmed_bits'] for r in rows),
                old_zero=sum(not r['online_committed_bits'] for r in rows),new_zero=sum(not r['total_confirmed_bits'] for r in rows))
    assert totals==dict(cases=64,online=336,terminal=56,total=392,old_zero=33,new_zero=25),totals
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'IMEC_ZERO_FORENSICS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'HISTORICAL_REPLAY.json').write_text(json.dumps(totals,indent=2)+'\n')
    print(json.dumps(totals))
    return totals

if __name__=='__main__': main()
