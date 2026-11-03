"""Read immutable workers and independent PPL scores into auditable tables."""
import argparse
import csv
import json
import math
from pathlib import Path

from stegatext_runtime.experiments.runner import read
from tools.controlled_comparison.prepare import OUT
from tools.controlled_comparison.statistics import summarize,paired_comparison,normalized_entropy
from tools.causal_audit.reward import tree_metrics


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def run_path(out,original):
    """Allow the evidence directory to move with a portable archive."""
    path=Path(original)
    local=out/'gpu'/path.parent.name/path.name
    if local.is_dir():
        return local
    if path.is_dir():
        return path
    raise FileNotFoundError(f'Run missing at {local} and {path}')


def collect(out):
    scores=read(out/'ppl/scores.json')['rows'] if (out/'ppl/scores.json').exists() else []
    lookup={(Path(r['run_directory']).name,r['case_id']):r for r in scores}
    rows=[]
    for job in json.loads((out/'runs.json').read_text()):
        folder=run_path(out,job['directory'])
        results=read(folder/'results.json')
        manifest=read(folder/'manifest.json')
        k=manifest['configuration']['source']['sampling']['top_k']
        for row in results['rows']:
            sender=read(folder/'sender'/(row['case_id']+'.json'))
            r=sender.get('result',{})
            trace=r.get('trace',[])
            ordinary=row['variant']['kind']=='ordinary'
            sampling=r.get('ordinary_sampling',[])
            supports=[t['positive_source_support'] for t in sampling] if ordinary else [n for t in trace for n in t.get('positive_source_support',[])]
            recorded_entropy=math.fsum(t['source_entropy_bits'] for t in sampling) if ordinary else math.fsum(t['group_entropy_bits'] for t in trace)
            has_vectors = all('group_probabilities' in t for t in trace)
            entropy=recorded_entropy if ordinary else math.fsum(
                normalized_entropy(t['group_probabilities']) if 'group_probabilities' in t else
                t['group_distribution']['entropy_bits_normalized'] for t in trace)
            coverage=[t['source_probability_mass_retained'] for t in trace if 'source_probability_mass_retained' in t]
            p=row.get('performance') or {}
            rp=row.get('receiver_performance') or {}
            score=lookup.get((folder.name,row['case_id']),{})
            h=r.get('coder_state',{}).get('entropies')
            online=r.get('online_committed_bits',r.get('bits_count',0))
            posterior=online+len(h)*row['variant'].get('coder_parameters',{}).get('block_size',6)-sum(h) if h is not None else online
            item=dict(run_directory=str(folder),run_relative_to_report=str(Path('gpu')/folder.parent.name/folder.name),
                case_id=row['case_id'],model=row['model'],top_k=k,
                prompt_id=row['prompt_id'],seed=row['seed'],variant=row['variant']['id'],
                method=row['variant'].get('method','ordinary'),coder=row['variant'].get('coder','none'),tau=row['variant'].get('tau'),
                verified_bits=row.get('strict_verified_payload_bits',0),sender_bits=r.get('bits_count'),
                online_bits=online,terminal_bits=r.get('terminal_extra_prefix_bits',0),
                bytes=(row.get('output') or {}).get('text_utf8_bytes'),evaluation_tokens=(row.get('output') or {}).get('evaluation_tokens'),
                group_entropy_sum=entropy if r else None,
                group_entropy_sum_recorded=recorded_entropy if r else None,
                entropy_normalization=('source_float_entropy_recorded' if ordinary else
                    'normalize_saved_coder_input_weights' if has_vectors else 'normalized_scalar_summary_and_saved_weights'),
                normalization_entropy_delta=entropy-recorded_entropy if r else None,
                group_probability_mass_error_max=max((abs((math.fsum(t['group_probabilities']) if 'group_probabilities' in t
                    else t['group_distribution']['weights_sum'])-1.) for t in trace),default=0),
                group_information_sum=None if ordinary or not r else math.fsum(-math.log2(
                    t['group_probabilities'][t['action']]/math.fsum(t['group_probabilities']) if 'group_probabilities' in t
                    else t['group_distribution']['selected_probability_normalized']) for t in trace),
                group_information_sum_recorded=None if ordinary or not r else math.fsum(t['selected_group_information_bits'] for t in trace),
                posterior_entropy_reduction=posterior,
                expected_coder_information_sum=math.fsum(t['coder']['expected_information_bits'] for t in trace) if h is not None else None,
                discop_ideal_expected_bits_sum=(math.fsum(tree_metrics(t['group_probabilities'])['expected_bits'] for t in trace)
                    if row['variant'].get('coder')=='discop_copy' and r and has_vectors else None),
                source_entropy_query_sum=entropy if ordinary else math.fsum(sum(t.get('source_entropy_bits',[])) for t in trace),
                source_queries=sum(t['model_queries'] for t in trace) if not ordinary else len(sampling),
                source_support_min=min(supports) if supports else None,source_support_max=max(supports) if supports else None,
                source_support_sum=sum(supports),source_support_queries=len(supports),
                sender_seconds=p.get('wall_seconds',0),receiver_seconds=rp.get('wall_seconds',0),
                physical_forwards=p.get('actual_model_forward_calls',0),reference_forwards=p.get('model_forwards',0),
                oracle_forwards=p.get('oracle_model_forward_calls',0),snapshot_restores=p.get('snapshot_restorations',0),
                planning_seconds=r.get('planning_seconds',0),coder_seconds=r.get('coder_seconds',0),
                peak_cuda_gib=p.get('peak_cuda_gib'),decisions=len(trace) if not ordinary else len(sampling),
                las_steps=sum(t.get('kernel')=='las' for t in trace),fullsync_steps=sum(t.get('kernel')=='fullsync' for t in trace),
                positive_u_steps=sum(t.get('U_post',0)>0 for t in trace),
                positive_u_las_steps=sum(t.get('kernel')=='las' and t.get('U_post',0)>0 for t in trace),
                positive_u_fullsync_steps=sum(t.get('kernel')=='fullsync' and t.get('U_post',0)>0 for t in trace),
                u_histogram={str(bound):sum(low<=t.get('U_post',0)<bound for t in trace if t.get('U_post',0)>0)
                             for low,bound in [(0,.25),(.25,.5),(.5,1.),(1.,2.),(2.,float('inf'))]},
                retained_mass_before_max=max((t.get('retained_mass_before',0) for t in trace),default=0),
                retained_mass_after_sum=math.fsum(t.get('retained_mass_after',0) for t in trace),
                retained_components_max=max((t.get('retained',0) for t in trace),default=0),
                strict_events=sum(t.get('strict_count',0)>0 for t in trace),
                max_u=max((t.get('U_post',0) for t in trace),default=None),coverage_values=coverage,
                coverage_information_tax=-math.fsum(math.log2(v) for v in coverage if v>0),
                sender_status=row['sender_status'],receiver_status=row.get('receiver_status'),
                exact=row['exact'],error=row.get('error'),terminal_reason=row['terminal_reason'],censored=row['censored'],
                natural_eos=r.get('natural_eos'),framing_status=row.get('framing_status'),
                source_generated_tokens=r.get('token_budget',{}).get('representative_generated_tokens'),
                text=r.get('text'),log_ppl=score.get('log_ppl'),ppl=score.get('ppl'),
                nll_total=score.get('nll_total'),ppl_tokens=score.get('evaluator_tokens'),
                ppl_metric=score.get('metric', 'legacy-retokenized-visible-text-base-v1' if score else None),
                native_token_sequence=r.get('native_token_sequence'),
                sampling_randomness=r.get('sampling_randomness'),
                protocol_randomness=r.get('protocol_randomness'), message_randomness=r.get('message_randomness'),
                evaluator_status=score.get('evaluator_status'),prefix_audit=r.get('prefix_audit'))
            rows.append(item)
    return rows


def analyze(out):
    rows=collect(out)
    groups={}
    for r in rows:
        groups.setdefault((r['model'],r['top_k'],r['variant']),[]).append(r)
    summaries=[]
    for (model,k,variant),group in sorted(groups.items()):
        first=group[0]
        summaries.append(dict(model=model,top_k=k,variant=variant,method=first['method'],coder=first['coder'],tau=first['tau'],**summarize(group)))
    comparisons=[]
    for (model,k,variant),group in sorted(groups.items()):
        if variant=='ordinary':
            continue
        first=group[0]
        short='imec' if first['coder']=='imec_prefix_exact' else 'discop'
        refs=['ordinary']
        if variant!=f'syncpool_{short}':
            refs.append(f'syncpool_{short}')
        if first['method']=='adaptive':
            refs.append(f'las_{short}')
        for reference in refs:
            result=paired_comparison(group,groups[(model,k,reference)])
            comparisons.append(dict(model=model,top_k=k,variant=variant,baseline=reference,**result))
    dump(out/'cases.json',rows)
    csv_write(out/'cases.csv',rows)
    dump(out/'aggregates.json',summaries)
    csv_write(out/'aggregates.csv',summaries)
    dump(out/'comparisons.json',comparisons)
    failures=[r for r in rows if r['error'] or r['exact'] is False]
    dump(out/'failures.json',failures)
    dump(out/'zero_payload.json',[r for r in rows if r['method']!='ordinary' and r['verified_bits']==0])
    totals=dict(cases=len(rows),stego_cases=sum(r['method']!='ordinary' for r in rows),
        prefix_exact=sum(r['exact'] is True for r in rows),failures=len(failures),
        zero_payload=sum(r['method']!='ordinary' and r['verified_bits']==0 for r in rows),
        successful_zero_payload=sum(r['method']!='ordinary' and r['exact'] is True and r['verified_bits']==0 for r in rows),
        natural_eos=sum(r['terminal_reason']=='natural_eos' for r in rows),
        token_limit=sum(r['terminal_reason']=='token_limit' for r in rows),
        verified_bits=sum(r['verified_bits'] for r in rows),
        ppl_scored=sum(r['evaluator_status']=='scored' for r in rows))
    dump(out/'summary.json',totals)
    print(json.dumps(totals,ensure_ascii=False,indent=2))
    return rows,summaries,comparisons


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    analyze(args.output.resolve())
