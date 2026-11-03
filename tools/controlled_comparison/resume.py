"""Continue missing cases while preserving successful AND failed observations.

The general framework's resume retries failed workers. A comparison must retain
the first observed failure, so this tool selects only missing cases
in a fresh worker process, leaving generation and recovery functions unchanged.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from stegatext_runtime.experiments.config import ROOT,resolve
from stegatext_runtime.experiments import runner


def finish_pending(config_path,directory):
    manifest=runner.read(directory/'manifest.json')
    execution=dict(configuration_sha256=runner.fingerprint(resolve(config_path)),
                   environment=runner.environment(),code=runner.code_state())
    assert manifest['execution']==execution,'Resume requires the exact original code/environment'
    retained={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
              for stage in ('sender','receiver') for p in (directory/stage).glob('*.json')}
    with (ROOT/'.cache/experiment_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model in manifest['configuration']['source']['models']:
            for stage in ('sender','receiver'):
                command=[sys.executable,'-B','-m','tools.controlled_comparison.resume','--worker',
                         '--run-dir',str(directory),'--model',model,'--stage',stage]
                with (directory/'logs'/f'{model}_{stage}.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(json.dumps(dict(model=model,stage=stage,status='finished',resumed=True)),flush=True)
        for name,digest in retained.items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
        rows=runner.summarize(directory)
    runner.write(directory/'resume_audit.json',dict(policy='preserve every existing final record, including failures',
        unchanged_files=retained,verified=True,configuration_sha256=execution['configuration_sha256']))
    return directory,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    manifest=runner.read(args.run_dir/'manifest.json')
    original_order=runner.ordered_cases
    pending=[]
    for case in original_order(manifest['configuration']):
        if case['model']!=args.model or (args.stage=='receiver' and case['variant']['kind']=='ordinary'):
            continue
        path=args.run_dir/args.stage/(case['id']+'.json')
        expected=runner.base_record(case,args.stage,manifest['execution'])
        public=args.run_dir/'public'/(case['id']+'.json')
        if args.stage=='receiver' and public.exists():
            expected['input_sha256']=runner.file_hash(public)
        runner.cached(path,expected)  # validates even a final error record
        if not path.exists():
            pending.append(case)
        elif args.stage=='sender':
            saved=runner.read(path)
            if saved['status']=='ok' and case['variant']['kind']!='ordinary':
                assert public.exists() and runner.file_hash(public)==saved['public_sha256']
                runner.read(public)
    runner.ordered_cases=lambda config:pending
    runner.worker(args.run_dir,args.model,args.stage)


if __name__=='__main__':
    main()
