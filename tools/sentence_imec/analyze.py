"""Offline exact coupling/posterior replay and complete-case smoke summaries."""
from __future__ import annotations

from collections import defaultdict, Counter
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import statistics

from stegatext_runtime.experiments.runner import read
from stegatext_runtime.experiments.config import ROOT
from stegatext_runtime.experiments.engine import message_for

OUT = ROOT/'reports/sentence_imec_20261018'


def ent(p):
    return -math.fsum(float(x)*(math.log2(x.numerator)-math.log2(x.denominator)) for x in p if x)


def unpack(rows,n=64):
    values = [F(0)]*n
    for i,p in rows:
        values[i] = F(p)
    return values


def replay(result, message, parameters):
    b,w = parameters['block_size'],parameters['window_size']
    window = [[F(1,1 << b)]*(1 << b) for _ in range(w)]
    head, committed, expected_info, group_h, observed_drop = 0,'',0.,0.,0.
    for step in result['trace']:
        c = step['coder']
        index = c['selected_window_index']
        assert head == c['head_before'] and c['selected_block'] == head+index
        h_before = [ent(p) for p in window]
        assert h_before[index] >= max(h_before)-1e-12
        p = unpack(c['posterior_before'],1 << b)
        assert p == window[index]
        q = [F(x) for x in c['group_probabilities']]
        assert sum(q) == sum(p) == 1
        rows,cols = [F(0)]*len(p),[F(0)]*len(q)
        selected = [F(0)]*len(p)
        joint = []
        action = step['action']
        target = int(message[(head+index)*b:(head+index+1)*b],2)
        true_pair_mass = F(0)
        for i,j,value in c['coupling_edges']:
            mass = F(value)
            assert mass > 0
            rows[i] += mass; cols[j] += mass
            joint.append(mass)
            if j == action:
                selected[i] += mass
                if i == target: true_pair_mass += mass
        assert rows == p and cols == q and q[action] > 0 and true_pair_mass > 0
        after = [x/q[action] for x in selected]
        assert after == unpack(c['posterior_after'],1 << b)
        window[index] = after
        observed_drop += sum(h_before)-sum(ent(p) for p in window)
        positive = [i for i,v in enumerate(window[0]) if v]
        out = ''
        if len(positive) == 1:
            out = format(positive[0],f'0{b}b')
            window.pop(0); window.append([F(1,1 << b)]*(1 << b)); head += 1
        assert c['committed_bits'] == out and c['head_after'] == head
        committed += out
        assert step['bits'] == len(committed)
        mi = ent(p)+ent(q)-ent(joint)
        assert abs(mi-c['expected_information_bits']) < 1e-9 and mi >= -1e-9
        group_h += ent(q); expected_info += mi
    final = result['coder_state']
    assert head == final['head_block'] and [unpack(p,1 << b) for p in final['beliefs']] == window
    assert committed == result['bits'] == message[:len(committed)]
    pending = b*w-sum(ent(p) for p in window)
    assert abs(observed_drop-(len(committed)+pending)) < 1e-8
    return dict(verified_steps=len(result['trace']), group_entropy_sum=group_h,
                expected_coupling_information_sum=expected_info,pending_information_bits=pending,
                posterior_information_bits=len(committed)+pending)


def mean(rows,key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def main(output=OUT):
    jobs = json.loads((output/'runs.json').read_text())
    rows, texts, replay_steps, fixtures = [],[],0,{}
    for job in jobs:
        directory = ROOT/job['directory']
        manifest = read(directory/'manifest.json')
        config = manifest['configuration']
        summary = read(directory/'results.json')
        for record in summary['rows']:
            case_id = record['case_id']
            s = read(directory/'sender'/f'{case_id}.json')
            result = s.get('result',{})
            ordinary = record['variant']['kind'] == 'ordinary'
            settings = s.get('conditions',{}).get('source',{}).get('generation_config',{})
            row = dict(experiment=job['config'], model=record['model'],case_id=case_id,
                prompt_id=record['prompt_id'],seed=record['seed'],variant=record['variant']['id'],ordinary=ordinary,
                top_k=settings.get('top_k'),top_p=settings.get('top_p'),temperature=settings.get('temperature'),
                sender_status=record['sender_status'],receiver_status=record['receiver_status'],exact=record['exact'],
                strict_prefix_verified=record.get('strict_prefix_verified'),
                verified_bits=record.get('strict_verified_payload_bits',0),submitted_bits=result.get('bits_count',0),
                utf8_bytes=len(result['text'].encode()) if result.get('text') is not None else None,
                evaluation_tokens=result.get('evaluation_token_count'),
                protocol_steps=result.get('steps'),
                terminal_reason=record['terminal_reason'],censored=record['censored'],
                sentence_goal_reached=result.get('sentence_goal_reached',False),
                completed_sentences=result.get('completed_sentences'),
                overshoot_utf8_bytes=result.get('overshoot_utf8_bytes'),
                sender_seconds=(record.get('performance') or {}).get('wall_seconds'),
                receiver_seconds=(record.get('receiver_performance') or {}).get('wall_seconds'),
                sender_forwards=(record.get('performance') or {}).get('model_forwards'),
                receiver_forwards=(record.get('receiver_performance') or {}).get('model_forwards'),
                coder_seconds=result.get('coder_seconds'),error=record.get('error'))
            row['peak_cuda_gib'] = (record.get('performance') or {}).get('peak_cuda_gib')
            supports = [x['positive_source_support'] for x in result.get('ordinary_sampling',[])] if ordinary else [n for t in result.get('trace',[]) for n in t.get('positive_source_support',[])]
            row.update(positive_support_mean=statistics.fmean(supports) if supports else None,
                positive_support_min=min(supports) if supports else None,positive_support_max=max(supports) if supports else None)
            if result and not ordinary:
                if record['seed'] not in fixtures:
                    fixtures[record['seed']] = message_for(config,record['seed'])
                row.update(replay(result,fixtures[record['seed']],record['variant']['coder_parameters']))
                beliefs = result['coder_state']['beliefs']
                drainable = 0
                for belief in beliefs:
                    if len(belief) != 1:
                        break
                    drainable += record['variant']['coder_parameters']['block_size']
                row.update(submitted_prefix=result['bits'],
                    unsubmitted_singleton_prefix_bits=drainable,
                    head_support_size=len(beliefs[0]),
                    head_block_entropy=result['coder_state']['entropies'][0],
                    singleton_window_blocks=sum(len(p) == 1 for p in beliefs),
                    retained_state_events=sum(t['retained'] > 0 for t in result['trace']),
                    multi_exact_group_events=sum(t.get('multi_exact_groups',0) > 0 for t in result['trace']),
                    maximum_retained_candidates=max(t['retained'] for t in result['trace']))
                replay_steps += row['verified_steps']
                receiver = read(directory/'receiver'/f'{case_id}.json')
                row['received_prefix'] = receiver.get('result',{}).get('bits')
                if receiver['status'] == 'ok':
                    assert receiver['result']['trace'] == result['trace']
                    assert receiver['result']['coder_state'] == result['coder_state']
                    assert receiver['result']['bits'] == result['bits']
                packet = read(directory/'public'/f'{case_id}.json')
                assert not ({'bits','trace','coder_state','token_ids'} & packet.keys())
                row['source_sha256'] = packet['source']['sha256']
            for performance in [record.get('performance'),record.get('receiver_performance')]:
                if performance:
                    assert performance['actual_model_forward_calls'] == performance['model_forwards']
                    assert performance['snapshot_oracle_checks'] == 16
            rows.append(row)
            texts.append(f'## {job["config"]} / {case_id}\n\n{result.get("text")}\n\nStop: {record["terminal_reason"]}; bytes past the sentence end: {row["overshoot_utf8_bytes"]}; confirmed bits: {row["verified_bits"]}.\n')
    for seed,bits in fixtures.items():
        (output/f'message_seed_{seed}.bits').write_text(bits+'\n')
    groups = defaultdict(list)
    for r in rows:
        groups[r['experiment'],r['model'],r['variant']].append(r)
    aggregates = []
    for (experiment,model,variant),group in sorted(groups.items()):
        bits = sum(r['verified_bits'] for r in group)
        size = sum(r['utf8_bytes'] for r in group) if all(r['utf8_bytes'] is not None for r in group) else None
        tokens = sum(r['evaluation_tokens'] for r in group) if all(r['evaluation_tokens'] is not None for r in group) else None
        seconds = sum(r['sender_seconds'] or 0 for r in group)
        item = dict(experiment=experiment,model=model,variant=variant,cases=len(group),
            verified_bits=bits,utf8_bytes=size,reliable_bpb=bits/size if size else None,
            evaluation_tokens=tokens,reliable_bpt=bits/tokens if tokens else None,
            strict_successes=sum(r['strict_prefix_verified'] is True for r in group),
            sentence_successes=sum(r['sentence_goal_reached'] for r in group),
            failed_cases=sum(r['sender_status'] != 'ok' or r['exact'] is False for r in group),
            zero_bit_cases=sum(r['verified_bits'] == 0 for r in group),
            positive_payload_cases=sum(r['verified_bits'] > 0 for r in group),
            terminal_reasons=dict(Counter(r['terminal_reason'] for r in group)),
            censored_cases=sum(r['censored'] is True for r in group),
            sender_seconds=seconds,verified_bits_per_second=bits/seconds if seconds else None,
            sender_forwards=sum(r['sender_forwards'] or 0 for r in group),
            receiver_forwards=sum(r['receiver_forwards'] or 0 for r in group),
            receiver_seconds=sum(r['receiver_seconds'] or 0 for r in group),
            protocol_steps=sum(r['protocol_steps'] or 0 for r in group),
            positive_support_min=min(r['positive_support_min'] for r in group if r['positive_support_min'] is not None),
            positive_support_max=max(r['positive_support_max'] for r in group if r['positive_support_max'] is not None),
            peak_cuda_gib=max(r['peak_cuda_gib'] for r in group if r['peak_cuda_gib'] is not None),
            **{'mean_'+key:mean(group,key) for key in ['group_entropy_sum','expected_coupling_information_sum',
                'pending_information_bits','posterior_information_bits','coder_seconds','overshoot_utf8_bytes',
                'positive_support_mean','singleton_window_blocks','head_block_entropy']})
        if not group[0]['ordinary']:
            for key in ['group_entropy_sum','expected_coupling_information_sum','pending_information_bits','posterior_information_bits']:
                item['total_'+key] = sum(r[key] for r in group)
                item[key+'_per_byte'] = item['total_'+key]/size if size else None
            item['coupling_information_to_group_entropy'] = item['total_expected_coupling_information_sum']/item['total_group_entropy_sum'] if item['total_group_entropy_sum'] else None
            item['retained_state_events'] = sum(r['retained_state_events'] for r in group)
            item['multi_exact_group_events'] = sum(r['multi_exact_group_events'] for r in group)
            item['cases_with_singleton_pending_blocks'] = sum(r['singleton_window_blocks'] > 0 for r in group)
            item['unsubmitted_singleton_prefix_bits'] = sum(r['unsubmitted_singleton_prefix_bits'] for r in group)
            item['end_to_end_bits_per_second'] = bits/(seconds+item['receiver_seconds'])
        aggregates.append(item)
    pairs = []
    for experiment,model in sorted({(r['experiment'],r['model']) for r in rows}):
        a = next(x for x in aggregates if (x['experiment'],x['model'],x['variant']) == (experiment,model,'syncpool_imec'))
        b = next(x for x in aggregates if (x['experiment'],x['model'],x['variant']) == (experiment,model,'las_imec'))
        pair = dict(experiment=experiment,model=model,syncpool_bpb=a['reliable_bpb'],las_bpb=b['reliable_bpb'],
                    delta_bpb=b['reliable_bpb']-a['reliable_bpb'] if a['reliable_bpb'] is not None and b['reliable_bpb'] is not None else None,
                    syncpool=a,las=b)
        pair['accounting_delta_posterior_information_per_byte'] = b['posterior_information_bits_per_byte']-a['posterior_information_bits_per_byte']
        pair['accounting_delta_pending_information_per_byte'] = b['pending_information_bits_per_byte']-a['pending_information_bits_per_byte']
        assert abs(pair['delta_bpb']-(pair['accounting_delta_posterior_information_per_byte']-pair['accounting_delta_pending_information_per_byte'])) < 1e-9
        matched = {}
        for r in rows:
            if r['experiment'] == experiment and r['model'] == model and not r['ordinary']:
                matched.setdefault((r['prompt_id'],r['seed']),[]).append(r)
        for batch in matched.values():
            assert len(batch) == 2 and batch[0]['source_sha256'] == batch[1]['source_sha256']
        pairs.append(pair)
    stego = [r for r in rows if not r['ordinary']]
    controlled = [r for r in stego if r['experiment'] != 'sentence_imec_base']
    summary = dict(cases=len(rows),stego_cases=len(stego),controlled_stego_cases=len(controlled),
        ordinary_cases=len(rows)-len(stego),strict_successes=sum(r['strict_prefix_verified'] is True for r in stego),
        sentence_successes=sum(r['sentence_goal_reached'] for r in rows),
        failures=[r['case_id'] for r in rows if r['sender_status'] != 'ok' or r['exact'] is False],
        censored=sum(r['censored'] is True for r in rows),
        offline_exact_posterior_replay_steps=replay_steps,aggregates=aggregates,pairs=pairs,
        go=all(r['strict_prefix_verified'] is True for r in controlled) and len(controlled) == 48
           and all(r['sentence_goal_reached'] for r in controlled),
        scope='Protocol smoke, not a statistically confirmed population capacity advantage',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fields = sorted({k for r in rows for k in r})
    with (output/'cases.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (output/'TEXTS.md').write_text('# All generated texts (unfiltered)\n\n'+'\n'.join(texts))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['aggregates','pairs']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',type=Path,default=OUT)
    main(parser.parse_args().directory.resolve())
