"""Rebuild one original case through the same unified sender/receiver entry."""
import argparse
import copy
import json
from pathlib import Path

from stegatext_runtime.experiments.config import resolve,cases
from stegatext_runtime.experiments.runner import read,run


def toml_value(value):
    if isinstance(value,str):
        return json.dumps(value,ensure_ascii=False)
    if isinstance(value,bool):
        return str(value).lower()
    if isinstance(value,(int,float)):
        return str(value)
    if isinstance(value,list):
        return '['+', '.join(toml_value(v) for v in value)+']'
    if isinstance(value,dict):
        return '{'+', '.join(f'{k}={toml_value(v)}' for k,v in value.items())+'}'
    raise ValueError(f'Unsupported TOML value {type(value)}')


def single_config(config,case_id):
    case=next(c for c in cases(config) if c['id']==case_id)
    result=copy.deepcopy(config)
    result.pop('model_revisions',None)
    result.get('randomness', {}).pop('sha256', None)
    result.get('message', {}).pop('sha256', None)
    if result.get('message', {}).get('mode') == 'file':
        result['message'].pop('bits', None)
    result['name']='single_'+config['name']
    result['source']['models']=[case['model']]
    result['source']['languages']={case['model']:config['source']['languages'][case['model']]}
    prompt=copy.deepcopy(case['prompt'])
    prompt.pop('sha256',None)
    result['inputs'].update(prompts=[prompt],seeds=[case['seed']])
    result['variants']=[case['variant']]
    return result


def write_config(path,config):
    lines=[f'schema_version = {config["schema_version"]}',f'name = {toml_value(config["name"])}']
    for section in ('source','inputs','message','randomness','stopping','execution'):
        if section in config:
            lines += ['',f'[{section}]']+[f'{k} = {toml_value(v)}' for k,v in config[section].items()]
    for variant in config['variants']:
        lines += ['','[[variants]]']+[f'{k} = {toml_value(v)}' for k,v in variant.items()]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config-only',action='store_true')
    args=parser.parse_args()
    original=args.run_dir.resolve()
    manifest=read(original/'manifest.json')
    out=args.output.resolve()
    config_path=out/'single_case.toml'
    write_config(config_path,single_config(manifest['configuration'],args.case_id))
    config=resolve(config_path)
    assert [c['id'] for c in cases(config)]==[args.case_id]
    if args.config_only:
        print(config_path)
        return
    directory,rows=run(config_path,out/'gpu')
    before=read(original/'sender'/(args.case_id+'.json'))
    after=read(directory/'sender'/(args.case_id+'.json'))
    checks=dict(sender_status=before['status']==after['status'],sender_error=before.get('error')==after.get('error'))
    if before['status']==after['status']=='ok':
        fields=('text','visible_bytes_hex','bits','natural_eos','state_sha256','trace','coder_state','token_budget','terminal_reason',
                'token_ids','native_token_sequence','sampling_randomness','protocol_randomness','message_randomness')
        checks.update({k:before['result'].get(k)==after['result'].get(k) for k in fields})
    old_rows=read(original/'results.json')['rows']
    old=next(r for r in old_rows if r['case_id']==args.case_id)
    checks['recovery_outcome']=old['exact']==rows[0]['exact']
    checks['reported_error']=old.get('error')==rows[0].get('error')
    original_receiver=original/'receiver'/(args.case_id+'.json')
    if original_receiver.exists():
        old_receiver=read(original_receiver)
        new_receiver=read(directory/'receiver'/(args.case_id+'.json'))
        checks['receiver_status']=old_receiver['status']==new_receiver['status']
        checks['receiver_error']=old_receiver.get('error')==new_receiver.get('error')
    result=dict(case_id=args.case_id,original_run=str(original),reproduced_run=str(directory),checks=checks,
                passed=all(checks.values()),timing='not expected bitwise identical')
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
