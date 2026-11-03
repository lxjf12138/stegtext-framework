"""CPU-only coder verification and lossless restoration of compact evidence.

No hidden LLM history is inferred here. Frozen group probabilities and actions
are sufficient to reconstruct deterministic iMEC matrices and exact posteriors.
This verifies the coder contract, not the correctness of the planner's groups.
"""
import argparse
import base64
import copy
import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path

import torch

from stegatext_runtime.experiments.config import fingerprint
from stegatext_runtime.experiments.engine import coder,message_for
from stegatext_runtime.experiments.engine_v2 import shared_key
from stegatext_runtime.experiments.runner import read
from stegatext_runtime.experiments.recording import coder_trace
from round4_2_common import CounterRNG


COMPACT_KEY='_reconstructible_trace_v1'
KEEP_CODER=('selected_window_index','selected_block','head_before','group_entropy_bits',
            'coupling_entropy_bits','expected_information_bits','exact_marginals','action',
            'committed_bits','head_after','storage')


def serialized(record):
    return (json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def compact_record(record, probability_catalog=None):
    """Retain evidence and scalar audits; remove only deterministically derived data."""
    if any('group_distribution' in step for step in record.get('result',{}).get('trace',[])):
        # These records are already bounded and require a model/source replay.
        # Do not label them as CPU-reconstructible full-vector evidence.
        return copy.deepcopy(record)
    value=copy.deepcopy(record)
    found=False
    for step in value.get('result',{}).get('trace',[]):
        p=step.get('group_probabilities')
        if isinstance(p,list) and all(isinstance(v,float) for v in p):
            # Binary64 roundtrips every JSON float, including non-float32 values.
            # Keep the key's insertion position for byte-identical restoration.
            raw=struct.pack('<'+'d'*len(p),*p)
            encoded=base64.b64encode(raw).decode()
            if probability_catalog is None:
                step['group_probabilities']={'binary64le_base64':encoded}
            else:
                digest=hashlib.sha256(raw).hexdigest()
                assert probability_catalog.setdefault(digest,encoded)==encoded
                step['group_probabilities']={'binary64le_sha256':digest}
            found=True
        if 'coder' in step:
            keep=KEEP_CODER
            if 'group_probabilities' not in step:
                keep+=('group_probabilities',)
            step['coder']={k:v for k,v in step['coder'].items() if k in keep}
            if 'group_probabilities' not in step and probability_catalog is not None:
                q=step['coder']['group_probabilities']
                digest=hashlib.sha256(json.dumps(q,separators=(',',':')).encode()).hexdigest()
                assert probability_catalog.setdefault(digest,q)==q
                step['coder']['group_probabilities']={'rational_sha256':digest}
            found=True
    if found:
        value[COMPACT_KEY]=dict(original_file_sha256=hashlib.sha256(serialized(record)).hexdigest(),
                               instructions='python -B -m tools.controlled_comparison.replay --restore REPORT_ROOT')
    return value


def regenerate_record(record,variant,probability_catalog=None):
    value=copy.deepcopy(record)
    info=value.pop(COMPACT_KEY,None)
    c=coder(variant['coder'],'decode',variant.get('coder_parameters'))
    if hasattr(c, 'set_trace_policy'):
        c.set_trace_policy(max_full_actions=None)  # explicit legacy audit only
    for step in value.get('result',{}).get('trace',[]):
        p=step.get('group_probabilities')
        if isinstance(p,dict):
            if 'binary64le_sha256' in p:
                digest=p['binary64le_sha256']
                if probability_catalog is None:
                    raise ValueError('Restore requires the shared PROBABILITY_VECTORS.json catalog')
                raw=base64.b64decode(probability_catalog[digest],validate=True)
                assert hashlib.sha256(raw).hexdigest()==digest
            else:
                raw=base64.b64decode(p['binary64le_base64'],validate=True)
            step['group_probabilities']=list(struct.unpack('<'+'d'*(len(raw)//8),raw))
        if 'coder' not in step:
            continue
        if 'group_probabilities' in step:
            probabilities=torch.tensor(step['group_probabilities'],dtype=torch.float64)
        else:
            q=step['coder']['group_probabilities']
            if isinstance(q,dict):
                digest=q['rational_sha256']
                if probability_catalog is None:
                    raise ValueError('Restore requires the shared PROBABILITY_VECTORS.json catalog')
                q=probability_catalog[digest]
                assert hashlib.sha256(json.dumps(q,separators=(',',':')).encode()).hexdigest()==digest
                step['coder']['group_probabilities']=q
            probabilities=list(map(Fraction,q))
        c.decode(probabilities,step['action'],rng=None)
        # Large traces retain exact reconstruction hashes instead of repeated
        # Fraction matrices. Verify those hashes as well as the scalar audit.
        rebuilt = coder_trace(c.last_trace) if 'storage' in step['coder'] else c.last_trace
        assert all(rebuilt[k]==v for k,v in step['coder'].items())
        step['coder']=copy.deepcopy(rebuilt)
    if info:
        assert hashlib.sha256(serialized(value)).hexdigest()==info['original_file_sha256']
    digest=value.pop('record_sha256')
    assert fingerprint(value)==digest
    value['record_sha256']=digest
    return value


def verify_case(directory,row,config):
    ident=row['case_id']
    s=read(directory/'sender'/(ident+'.json'))
    if row['variant']['kind']=='ordinary' or s['status']!='ok':
        return dict(case_id=ident,status='not_a_successful_stego_sender')
    if any('group_distribution' in step for step in s['result']['trace']):
        return dict(case_id=ident,status='requires_frozen_source_replay',
            strict_channel_exact=row['exact'],
            reason='Large distributions are summaries; use the independent receiver and GPU reproduce entry')
    packet=read(directory/'public'/(ident+'.json'))
    key=shared_key(config,row['seed'],packet['nonce'])
    stream=message_for(config,row['seed'],context=packet['nonce'])
    variant=row['variant']
    enc=coder(variant['coder'],'encode',variant.get('coder_parameters'))
    dec=coder(variant['coder'],'decode',variant.get('coder_parameters'))
    for session in (enc, dec):
        if hasattr(session, 'set_trace_policy'):
            session.set_trace_policy(max_full_actions=None)
    offset=0
    online=[]
    for step in s['result']['trace']:
        probs=torch.tensor(step['group_probabilities'],dtype=torch.float64)
        bits,action=enc.encode(probs,stream[offset:],rng=CounterRNG(key,'coder',step['step']))
        recovered=dec.decode(probs,action,rng=CounterRNG(key,'coder',step['step']))
        assert action==step['action'] and recovered==bits
        offset+=len(bits)
        assert offset==step['bits']
        online.append(bits)
        if 'coder' in step:
            assert enc.last_trace==dec.last_trace
            saved = step['coder']
            expected = coder_trace(enc.last_trace) if saved.get('storage',{}).get('version') == 'reconstructible-imec-trace-v1' else enc.last_trace
            assert expected==saved
    tail=enc.finalize_confirmed_prefix() if hasattr(enc,'finalize_confirmed_prefix') else ''
    if hasattr(dec,'finalize_confirmed_prefix'):
        assert tail==dec.finalize_confirmed_prefix()
    delivered=''.join(online)+tail
    assert delivered==s['result']['bits']==stream[:len(delivered)]
    if hasattr(enc,'state'):
        assert enc.state()==dec.state()==s['result']['coder_state']
    receiver=read(directory/'receiver'/(ident+'.json'))
    if receiver['status']=='ok':
        # A Direct control may decode another token history; report its mismatch
        # separately instead of treating coder replay as strict channel recovery.
        independent_bits_match=receiver['result']['bits']==delivered
    else:
        independent_bits_match=False
    return dict(case_id=ident,status='coder_exact',online_bits=offset,terminal_bits=len(tail),
                sender_confirmed_bits=len(delivered),strict_channel_exact=row['exact'],
                independent_receiver_bits_match=independent_bits_match)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reports/controlled_comparison_20261019'))
    parser.add_argument('--restore',type=Path)
    args=parser.parse_args()
    out=(args.restore or args.output).resolve()
    torch.set_num_threads(1)
    if args.restore:
        count=0
        catalog_path=out/'PROBABILITY_VECTORS.json'
        catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else None
        for manifest_path in sorted(out.parent.rglob('manifest.json')):
            if manifest_path.parent.parent.parent.name!='gpu':
                continue
            manifest=read(manifest_path)
            variants={v['id']:v for v in manifest['configuration']['variants']}
            for stage in ('sender','receiver'):
                for path in sorted((manifest_path.parent/stage).glob('*.json')):
                    record=json.loads(path.read_text())
                    if COMPACT_KEY not in record:
                        continue
                    variant=variants[record['case_id'].split('__')[-1]]
                    value=regenerate_record(record,variant,catalog)
                    path.write_bytes(serialized(value))
                    count+=1
        print(json.dumps(dict(restored_original_worker_files=count,sha256_verified=True)),flush=True)
        return
    from tools.controlled_comparison.analyze import run_path
    rows=[]
    for job in json.loads((out/'runs.json').read_text()):
        folder=run_path(out,job['directory'])
        config=read(folder/'manifest.json')['configuration']
        for row in read(folder/'results.json')['rows']:
            rows.append(dict(run=folder.name,**verify_case(folder,row,config)))
            if len(rows)%50==0:
                print(json.dumps(dict(verified_or_explicitly_skipped=len(rows))),flush=True)
    value=dict(rows=rows,cases=len(rows),coder_verified=sum(r['status']=='coder_exact' for r in rows),
               scope='Coder only; finite-source CPU oracle and independent GPU receivers separately validate channel')
    (out/'CODER_REPLAY.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
