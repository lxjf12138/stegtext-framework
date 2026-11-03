"""Serial T1 smoke with CPU gate and immutable unified-worker evidence."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from stegatext_runtime.experiments.config import ROOT,resolve,cases
from stegatext_runtime.experiments.runner import run,read,code_state,environment

GATE=ROOT/'reports/framework_contract_20261018'
CONFIGS=['sentence_T1_coder_audit','sentence_T1_coder_audit_k128','sentence_T1_coder_audit_k256']

def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=GATE)
    parser.add_argument('--resume-evidence',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    passed=re.search(r'(?m)^(\d+) passed(?:,.*)? in ',(GATE/'audit/cpu_full.log').read_text())
    assert passed and int(passed[1])>=214
    assert json.loads((GATE/'audit/HISTORICAL_REPLAY.json').read_text())['total']==392
    assert json.loads((GATE/'audit/INDEPENDENT_RECEIVER.json').read_text())['exact']
    plans={n:resolve(ROOT/f'configs/experiments/{n}.toml') for n in CONFIGS}
    assert sum(len(list(cases(v))) for v in plans.values())==96
    if (out/'preflight.json').exists():
        if not args.resume_evidence: raise RuntimeError('Preserve frozen attempt; use --resume-evidence or --output NEW_DIR')
        assert json.loads((out/'preflight.json').read_text())['configurations']==plans
    else:
        write_json(out/'preflight.json',dict(configurations=plans,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            cpu_gate='214 project tests; 15 oracle contracts; 64 replay; 320 frozen-stream trials',
            scope='operational stopping comparison, no stopped-cover-law or capacity dominance claim'))
    jobs=[]
    for n in CONFIGS:
        existing=list((out/'gpu'/plans[n]['name']).glob('*/results.json'))
        if existing:
            assert args.resume_evidence and len(existing)==1
            directory=existing[0].parent
            manifest=read(directory/'manifest.json')
            assert manifest['configuration']==plans[n]
            assert manifest['execution']['code']==code_state() and manifest['execution']['environment']==environment()
            rows=read(existing[0])['rows']
            assert len(rows)==len(list(cases(plans[n])))
        else:
            directory,rows=run(ROOT/f'configs/experiments/{n}.toml',out/'gpu')
        jobs.append(dict(config=n,directory=directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else str(directory),
            cases=len(rows),exact=sum(r['exact'] is True for r in rows),errors=sum(r['exact'] is not True for r in rows)))
        write_json(out/'runs.json',jobs)
        if jobs[-1]['errors']: raise RuntimeError('P2 failed: preserve first failing state; stop next conditions')
    write_json(out/'completion.json',dict(completed=True,jobs=jobs))

if __name__=='__main__': main()
