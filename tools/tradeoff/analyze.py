"""Auditable, paired development estimates. No row filtering by success."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from stegatext_runtime.experiments.runner import read, file_hash, code_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/tradeoff_20261017'
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261017
LAMBDAS = [0., .001, .01, .1, 1.]


def dump(name, value):
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {k:finite(v) for k,v in x.items()}
        if isinstance(x, list): return [finite(v) for v in x]
        return x
    (OUT/name).write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summary(rows):
    n = len(rows)
    missing = sum(r['bytes'] is None for r in rows)
    bits = sum(r['verified_bits'] for r in rows)
    length = None if missing else sum(r['bytes'] for r in rows)
    tx = sum(r['sender_seconds'] for r in rows)
    rx = sum(r['receiver_seconds'] for r in rows)
    scored = [r for r in rows if r.get('evaluator_status') == 'scored']
    nt = sum(r['evaluator_tokens'] for r in scored)
    nll = sum(r['nll_total'] for r in scored)
    return dict(cases=n, bits=bits, bytes=length, sender_seconds=tx, receiver_seconds=rx,
                bpb=bits/length if length else None, sender_bps=bits/tx if tx else None,
                end_to_end_bps=bits/(tx+rx) if tx+rx else None,
                sender_seconds_per_bit=tx/bits if bits else None,
                end_to_end_seconds_per_bit=(tx+rx)/bits if bits else None,
                bytes_per_bit=length/bits if bits and length is not None else None,
                sender_forwards=sum(r['sender_forwards'] for r in rows),
                sender_oracle_forwards=sum(r.get('sender_oracle_forwards',0) for r in rows),
                sender_queries=sum(r['sender_queries'] for r in rows),
                forwards_per_bit=sum(r['sender_forwards'] for r in rows)/bits if bits else None,
                model_forwards_per_query=sum(r['sender_forwards'] for r in rows)/sum(r['sender_queries'] for r in rows) if sum(r['sender_queries'] for r in rows) else None,
                exact=sum(r['exact'] is True for r in rows), errors=sum(r['error'] is not None for r in rows),
                zero_payload=sum(r['verified_bits'] == 0 for r in rows), censored=sum(r['censored'] is True for r in rows),
                natural_eos=sum(r['terminal_reason'] == 'natural_eos' for r in rows), invalid_or_missing_text=missing,
                mean_bytes=length/n if length is not None else None,
                las_kernel_steps=sum(r['las_kernel_steps'] for r in rows),
                fullsync_kernel_steps=sum(r['fullsync_kernel_steps'] for r in rows),
                positive_u_steps=sum(r['positive_u_steps'] for r in rows) if all(r['positive_u_steps'] is not None for r in rows) else None,
                decisions=sum(r['decisions'] for r in rows),
                strict_events=sum(r['strict_events'] for r in rows) if all(r['strict_events'] is not None for r in rows) else None,
                group_entropy_sum=sum(r['group_entropy_sum'] for r in rows),
                corpus_ppl=math.exp(nll/nt) if nt else None,
                mean_log_ppl=float(np.mean([r['log_ppl'] for r in scored])) if scored else None,
                ppl_scored=len(scored), ppl_missing=n-len(scored),
                roundtrip_changed=sum(bool(r.get('roundtrip_text_changed')) for r in scored))


def comparison(a, b):
    sa, sb = summary(a), summary(b)
    out = {}
    for metric in ['bpb', 'sender_bps', 'end_to_end_bps', 'forwards_per_bit', 'mean_bytes']:
        va, vb = sa[metric], sb[metric]
        out[metric+'_ratio'] = va/vb if va is not None and vb else None
    out['log_ppl_difference'] = sa['mean_log_ppl']-sb['mean_log_ppl'] if sa['mean_log_ppl'] is not None and sb['mean_log_ppl'] is not None else None
    for mode in ['sender', 'end_to_end']:
        ta, tb = sa[mode+'_seconds_per_bit'], sb[mode+'_seconds_per_bit']
        da, db = sa['bytes_per_bit'], sb['bytes_per_bit']
        if None in (ta, tb, da, db):
            out[mode+'_break_even_seconds_per_byte'] = None
        elif db > da and ta > tb:
            out[mode+'_break_even_seconds_per_byte'] = (ta-tb)/(db-da)
        else:
            out[mode+'_break_even_seconds_per_byte'] = None
    return out


def bootstrap(a, b):
    pa = defaultdict(list); pb = defaultdict(list)
    for r in a: pa[r['prompt_id']].append(r)
    for r in b: pb[r['prompt_id']].append(r)
    if set(pa) != set(pb):
        raise ValueError('Unpaired prompts')
    for p in pa:
        if sorted(r['seed'] for r in pa[p]) != sorted(r['seed'] for r in pb[p]):
            raise ValueError('Unpaired seeds')
    ps = sorted(pa)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Aggregate sufficient statistics, preserving all seeds in each prompt.
    def sums(group):
        scored = [r for r in group if r.get('log_ppl') is not None]
        return [sum(r['verified_bits'] for r in group), sum(r['bytes'] or 0 for r in group),
                sum(r['sender_seconds'] for r in group), sum(r['receiver_seconds'] for r in group),
                sum(r['sender_forwards'] for r in group), sum(r['log_ppl'] for r in scored), len(scored), len(group)]
    aa = np.array([sums(pa[p]) for p in ps], dtype=float)
    bb = np.array([sums(pb[p]) for p in ps], dtype=float)
    ids = rng.integers(0, len(ps), size=(BOOTSTRAPS, len(ps)))
    aa, bb = aa[ids].sum(axis=1), bb[ids].sum(axis=1)
    def ratios(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return dict(bpb=x[:, 0]/x[:, 1], sender_bps=x[:, 0]/x[:, 2],
                        end_to_end_bps=x[:, 0]/(x[:, 2]+x[:, 3]),
                        forwards_per_bit=x[:, 4]/x[:, 0], mean_bytes=x[:, 1]/x[:, 7])
    ra, rb = ratios(aa), ratios(bb)
    draws = {metric+'_ratio': ra[metric]/rb[metric] for metric in ra}
    draws['log_ppl_difference'] = aa[:, 5]/aa[:, 6]-bb[:, 5]/bb[:, 6]
    intervals = {}
    for key, values in draws.items():
        if key == 'bpb_ratio' and any(r['bytes'] is None for r in a+b):
            continue
        values = values[np.isfinite(values)]
        intervals[key] = dict(ci95=np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                              finite_draws=len(values))
    return dict(point=comparison(a, b), intervals=intervals, prompt_clusters=len(ps),
                draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
                interpretation='Development, marginal percentile intervals; no multiplicity adjustment or equivalence claim')


def cost(s, lam, mode):
    if s['bytes_per_bit'] is None or s[mode+'_seconds_per_bit'] is None:
        return math.inf
    return s[mode+'_seconds_per_bit']+lam*s['bytes_per_bit']


def lower_envelope(summaries, mode):
    """Exact breakpoints for the finite empirical affine-cost candidates."""
    lines={m:(s[mode+'_seconds_per_bit'],s['bytes_per_bit']) for m,s in summaries.items()
           if m!='ordinary' and s[mode+'_seconds_per_bit'] is not None and s['bytes_per_bit'] is not None}
    if not lines:return []
    cuts={0.}
    names=sorted(lines)
    for i,m in enumerate(names):
        for n in names[i+1:]:
            a,b=lines[m];c,d=lines[n]
            if b!=d:
                x=(c-a)/(b-d)
                if x>0 and math.isfinite(x):cuts.add(x)
    cuts=sorted(cuts)
    intervals=[]
    for i,lo in enumerate(cuts):
        hi=cuts[i+1] if i+1<len(cuts) else None
        x=(lo+hi)/2 if hi is not None else lo+max(1.,lo)
        values={m:a+b*x for m,(a,b) in lines.items()}
        best=min(values.values())
        selected=sorted(m for m,v in values.items() if abs(v-best)<=1e-10*max(1.,abs(best)))
        if intervals and intervals[-1]['methods']==selected:
            intervals[-1]['lambda_to']=hi
        else:intervals.append(dict(lambda_from=lo,lambda_to=hi,methods=selected))
    return intervals


def csv_write(name, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archived-raw',action='store_true',help='Read bundled raw evidence without original machine paths; keep all runtime hash checks')
    args=parser.parse_args()
    ledger = json.loads((OUT/'run_ledger.json').read_text())
    scorefile = OUT/'ppl/scores.json'
    scores = read(scorefile)['rows'] if scorefile.exists() else []
    score_lookup = {(r['run_directory'], r['case_id']): r for r in scores}
    rows = []
    checks = dict(all_planned_cases=True, hash_records=True, source_consistent=True,
                  independent_receivers=True, no_trace_in_public=True, complete_replay=True,
                  termination_consistent=True, runtime_unchanged=True)
    source_laws = defaultdict(set)
    entropy_samples = defaultdict(list)
    texts = ['# All development texts (every sample kept)', '']
    for stage in ledger:
        if 'run_directory' not in stage or not stage['run_directory']: continue
        original_directory=Path(stage['run_directory'])
        directory=(OUT/'raw'/stage['stage'] if args.archived_raw or not original_directory.exists() else original_directory)
        manifest, result = read(directory/'manifest.json'), read(directory/'results.json')
        expected = set(manifest['plan']['case_ids'])
        checks['all_planned_cases'] &= set(r['case_id'] for r in result['rows']) == expected and len(result['rows']) == len(expected)
        current = code_state()
        checks['runtime_unchanged'] &= manifest['execution']['code'] == current
        for r in result['rows']:
            cid = r['case_id']
            sender = read(directory/'sender'/(cid+'.json'))
            value = sender.get('result', {})
            trace = value.get('trace', [])
            entropy_samples[(stage['stage'], r['model'], r['variant']['id'])].extend(
                t['U_post'] for t in trace if 'U_post' in t)
            ordinary = r['variant']['kind'] == 'ordinary'
            source_laws[r['model']].add(json.dumps((r.get('conditions') or {}).get('source'), sort_keys=True))
            if not ordinary and sender['status'] == 'ok':
                public = read(directory/'public'/(cid+'.json'))
                receiver = read(directory/'receiver'/(cid+'.json'))
                checks['independent_receivers'] &= sender['pid'] != receiver['pid'] and receiver.get('input_sha256') == file_hash(directory/'public'/(cid+'.json'))
                checks['independent_receivers'] &= sender.get('public_sha256') == file_hash(directory/'public'/(cid+'.json'))
                checks['no_trace_in_public'] &= not bool({'trace', 'bits', 'token_ids', 'state_sha256'} & public.keys())
                checks['complete_replay'] &= r['exact'] is True and receiver.get('result', {}).get('receiver_consumed') is True
                checks['termination_consistent'] &= public['termination'] == {k:value.get(k) for k in public['termination']} == {k:receiver.get('result',{}).get(k) for k in public['termination']}
            elif not ordinary:
                checks['complete_replay'] = False
            counts = Counter(t.get('kernel') for t in trace)
            out = dict(stage=stage['stage'], run_directory=str(directory), case_id=cid, model=r['model'],
                       kv_cache=manifest['configuration']['source']['kv_cache'],
                       prompt_id=r['prompt_id'], seed=r['seed'], method=r['variant']['id'], tau=r['variant'].get('tau'),
                       exact=r['exact'], error=r.get('error'), sender_status=r['sender_status'], receiver_status=r['receiver_status'],
                       terminal_reason=r['terminal_reason'], censored=r['censored'], bytes=(r.get('output') or {}).get('text_utf8_bytes'),
                       measured_prefix_bits=r.get('confirmed_prefix_bits',0),
                       verified_bits=r.get('confirmed_prefix_bits', 0) if r['exact'] is True else 0,
                       sender_seconds=(r.get('performance') or {}).get('wall_seconds', 0),
                       receiver_seconds=(r.get('receiver_performance') or {}).get('wall_seconds', 0),
                       sender_forwards=(r.get('performance') or {}).get('actual_model_forward_calls', (r.get('performance') or {}).get('model_forwards', 0)),
                       receiver_forwards=(r.get('receiver_performance') or {}).get('actual_model_forward_calls', (r.get('receiver_performance') or {}).get('model_forwards', 0)),
                       sender_reference_forwards=(r.get('performance') or {}).get('model_forwards', 0),
                       sender_oracle_forwards=(r.get('performance') or {}).get('oracle_model_forward_calls', 0),
                       sender_queries=sum(t.get('model_queries', 0) for t in trace) if trace else len(value.get('token_ids', [])),
                       decisions=len(trace), las_kernel_steps=counts['las'], fullsync_kernel_steps=counts['fullsync'],
                       group_entropy_sum=sum(t.get('group_entropy_bits', 0) for t in trace),
                       positive_u_steps=sum(t['U_post']>1e-10 for t in trace) if trace and all('U_post' in t for t in trace) else None,
                       strict_events=sum(t['strict_count']>0 for t in trace) if trace and all('strict_count' in t for t in trace) else None,
                       peak_u=max(t['U_post'] for t in trace) if trace and all('U_post' in t for t in trace) else None,
                       peak_retained_mass=max((t.get('retained_mass',0) for t in trace),default=0),
                       peak_frontier=max((t.get('retained',0) for t in trace),default=0),
                       min_threshold_margin=min((abs(t.get('U_post',0)-r['variant']['tau']) for t in trace),default=None) if r['variant'].get('tau') is not None else None,
                       near_threshold_steps=sum(abs(t.get('U_post',0)-r['variant']['tau'])<1e-6 for t in trace) if r['variant'].get('tau') is not None else None,
                       planning_seconds=value.get('planning_seconds'), coder_seconds=value.get('coder_seconds'),
                       peak_cuda_gib=(r.get('performance') or {}).get('peak_cuda_gib'))
            score = score_lookup.get((str(original_directory), cid), {})
            for k in ['evaluator_status','evaluator_tokens','nll_total','log_ppl','ppl','nll_per_byte','roundtrip_text_changed']:
                out[k] = score.get(k)
            for k in ['repeated_12char_fraction','substantive_complete_sentences','sentence_end','drift_markers']:
                out[k] = (r.get('quality_screen') or {}).get(k)
            rows.append(out)
            texts += [f'## {stage["stage"]} / {cid}', '',
                      f"terminal={r['terminal_reason']}; censored={r['censored']}; verified_bits={out['verified_bits']}; error={r.get('error')}", '',
                      value.get('text') if value.get('text') is not None else '[no valid text; see the raw record]', '']
    checks['source_consistent'] = all(len(v)==1 for v in source_laws.values())
    csv_write('cases.csv', rows)
    dump('cases.json', rows)
    (OUT/'TEXTS.md').write_text('\n'.join(texts)+'\n')
    core = [r for r in rows if r['stage']=='pilot_tradeoff_v2']
    entropy_diagnostics = []
    for (stage, model, method), values in entropy_samples.items():
        if not values: continue
        positive = [u for u in values if u > 1e-10]
        entropy_diagnostics.append(dict(stage=stage,model=model,method=method,
            decisions_with_u=len(values),positive_u_decisions=len(positive),
            positive_u_fraction=len(positive)/len(values),max_u=max(values),
            positive_u_quantiles=np.quantile(positive,[.1,.5,.9]).tolist() if positive else None,
            observed_fractions_above_tau={str(tau):sum(u>=tau for u in values)/len(values) for tau in [.25,.5,1.,2.]},
            interpretation='Descriptive distribution on this policy trajectory; not a counterfactual rollout or payload bound'))
    paired_behavior = []
    for adaptive in [r for r in core if r['method']=='adaptive']:
        las=next(r for r in core if (r['model'],r['prompt_id'],r['seed'],r['method'])==
                 (adaptive['model'],adaptive['prompt_id'],adaptive['seed'],'las'))
        a=read(Path(adaptive['run_directory'])/'sender'/(adaptive['case_id']+'.json')).get('result',{})
        b=read(Path(las['run_directory'])/'sender'/(las['case_id']+'.json')).get('result',{})
        paired_behavior.append(dict(model=adaptive['model'],prompt_id=adaptive['prompt_id'],seed=adaptive['seed'],
            same_text=a.get('text') is not None and a.get('text')==b.get('text'),
            same_bits=a.get('bits') is not None and a.get('bits')==b.get('bits'),
            same_state=a.get('state_sha256') is not None and a.get('state_sha256')==b.get('state_sha256')))
    dump('threshold_diagnostics.json',dict(entropy=entropy_diagnostics,las_adaptive_pairs=paired_behavior))
    behavior_pairs=[]
    for method, reference_stage, reference_method in [('adaptive_025','pilot_tradeoff_v2','las'),
                                                       ('adaptive_200','threshold_tradeoff_v2','fullsync')]:
        for r in [x for x in rows if x['stage']=='threshold_tradeoff_v2' and x['method']==method]:
            reference=next(x for x in rows if x['stage']==reference_stage and
                           (x['model'],x['prompt_id'],x['seed'],x['method'])==
                           (r['model'],r['prompt_id'],r['seed'],reference_method))
            a=read(Path(r['run_directory'])/'sender'/(r['case_id']+'.json')).get('result',{})
            b=read(Path(reference['run_directory'])/'sender'/(reference['case_id']+'.json')).get('result',{})
            fields=['text','bits','state_sha256','terminal_reason','censored']
            if method=='adaptive_200': fields.append('trace')
            behavior_pairs.append(dict(model=r['model'],case_id=r['case_id'],method=method,reference=reference_method,
                fields=fields,matches=all(k in a and k in b and a[k]==b[k] for k in fields)))
    dump('threshold_behavior_pairs.json',behavior_pairs)
    sub = [r for r in rows if r['stage']=='threshold_tradeoff_v2' or
           (r['stage']=='pilot_tradeoff_v2' and r['prompt_id'].endswith(('01','03','05','07')))]
    aggregates, comparisons, cv, frontier, recommendations, envelopes = [], [], [], [], [], []
    cohorts=[('main192',core),('threshold_matched',sub),
             ('main_steady_sensitivity',[r for r in core if not r['prompt_id'].endswith('01')]),
             ('threshold_steady_sensitivity',[r for r in sub if not r['prompt_id'].endswith('01')])]
    canonical_sp=[r for r in rows if r['stage']=='syncpool_canonical_cost_v2']
    engineering_main=[r for r in core if r['method']!='syncpool']+canonical_sp
    engineering_sub=[r for r in sub if r['method']!='syncpool']+[r for r in canonical_sp if r['prompt_id'].endswith(('01','03','05','07'))]
    cohorts += [('engineering_main',engineering_main),('engineering_threshold',engineering_sub),
                ('engineering_main_steady',[r for r in engineering_main if not r['prompt_id'].endswith('01')]),
                ('engineering_threshold_steady',[r for r in engineering_sub if not r['prompt_id'].endswith('01')])]
    for cohort, rr in cohorts:
        for model in sorted({r['model'] for r in rr}):
            mm = [r for r in rr if r['model']==model]
            groups = {method:[r for r in mm if r['method']==method] for method in sorted({r['method'] for r in mm})}
            ss = {method:summary(group) for method, group in groups.items()}
            for method, s in ss.items():
                aggregates.append(dict(cohort=cohort, model=model, method=method, **s))
                if method == 'ordinary': continue
                dominated = [other for other, v in ss.items() if other!='ordinary' and other!=method and
                             None not in (s['bpb'],s['sender_bps'],v['bpb'],v['sender_bps']) and
                             v['bpb']>=s['bpb'] and v['sender_bps']>=s['sender_bps'] and
                             (v['bpb']>s['bpb'] or v['sender_bps']>s['sender_bps'])]
                frontier.append(dict(cohort=cohort, model=model, method=method, dominated_by=dominated, on_frontier=not dominated))
                for baseline in ['syncpool', 'las']:
                    if method == baseline: continue
                    comparisons.append(dict(cohort=cohort,model=model,method=method,baseline=baseline,**bootstrap(groups[method],groups[baseline])))
            for mode in ['sender','end_to_end']:
                envelopes.append(dict(cohort=cohort,model=model,timing=mode,
                                      intervals=lower_envelope(ss,mode),
                                      scope='Finite measured candidates only; point estimates, no statistical optimality claim'))
                for lam in LAMBDAS:
                    candidates = {m:cost(s,lam,mode) for m,s in ss.items() if m!='ordinary'}
                    best = min(candidates,key=candidates.get)
                    recommendations.append(dict(cohort=cohort,model=model,timing=mode,lambda_seconds_per_byte=lam,
                        selected=best,cost_seconds_per_bit=candidates[best],costs=candidates))
            if cohort in ('threshold_matched','engineering_threshold'):
                # Four held-out prompt domains, all 3 seeds held out together.
                candidates = ['adaptive_025','adaptive','adaptive_100','adaptive_200']
                for mode in ['sender','end_to_end']:
                    for lam in LAMBDAS:
                        selected_rows, folds = [], []
                        for pid in sorted({r['prompt_id'] for r in mm}):
                            train = {m:summary([r for r in groups[m] if r['prompt_id']!=pid]) for m in candidates}
                            best = min(candidates,key=lambda m:cost(train[m],lam,mode))
                            held = [r for r in groups[best] if r['prompt_id']==pid]
                            selected_rows.extend(held)
                            folds.append(dict(held_prompt=pid,selected=best))
                        selected = summary(selected_rows)
                        cv.append(dict(cohort=cohort,model=model,timing=mode,lambda_seconds_per_byte=lam,folds=folds,summary=selected,
                            cost=cost(selected,lam,mode),vs_las_cost_ratio=cost(selected,lam,mode)/cost(ss['las'],lam,mode),
                            vs_syncpool_cost_ratio=cost(selected,lam,mode)/cost(ss['syncpool'],lam,mode)))
    # Real tau=0 output/state/trace endpoint, including every public byte.
    endpoints = []
    for r in [r for r in rows if r['stage']=='endpoint_tradeoff_v2']:
        reference = next(x for x in core if (x['model'],x['prompt_id'],x['seed'],x['method'])==(r['model'],r['prompt_id'],r['seed'],'las'))
        left = read(Path(r['run_directory'])/'sender'/(r['case_id']+'.json'))['result']
        right = read(Path(reference['run_directory'])/'sender'/(reference['case_id']+'.json'))['result']
        fields = ['text','visible_bytes_hex','bits','trace','state_sha256','terminal_reason','censored']
        endpoints.append(dict(case_id=r['case_id'],matches=all(left[k]==right[k] for k in fields),fields=fields))
    checks['adaptive_zero_equals_las'] = len(endpoints)==4 and all(x['matches'] for x in endpoints)
    checks['all_ppl_scored'] = len(scores)==len(rows) and all(r['evaluator_status']=='scored' for r in rows)
    checks['source_exact_cache_conformance'] = json.loads((OUT/'cache_conformance.json').read_text())['all_passed']
    checks['frozen_sources_unchanged'] = not json.loads((OUT/'frozen_source_check.json').read_text())['changed_or_missing']
    finite_evidence = json.loads((OUT/'finite_threshold_evidence.json').read_text())
    checks['finite_threshold_oracles_passed'] = finite_evidence['all_passed'] and len(finite_evidence['cases'])==18
    checks['original_package_selftests_passed'] = 'all starter selftests passed' in (OUT/'frozen_selftest.log').read_text()
    expected_counts={'pilot_tradeoff_v2':192,'threshold_tradeoff_v2':96,'endpoint_tradeoff_v2':4,'syncpool_canonical_cost_v2':48}
    checks['gpu_matrix_complete'] = Counter(r['stage'] for r in rows)==Counter(expected_counts)
    checks['all_gpu_status_ok'] = all(r['sender_status']=='ok' and r['exact'] is not False for r in rows)
    suites=list(ElementTree.parse(OUT/'cpu_junit.xml').getroot().iter('testsuite'))
    cpu_count=sum(int(s.get('tests','0')) for s in suites)
    checks['cpu_regressions_passed'] = cpu_count>=137 and all(int(s.get('errors','0'))==int(s.get('failures','0'))==0 for s in suites)
    sp_pairs=[]
    for r in canonical_sp:
        reference=next(x for x in core if x['case_id']==r['case_id'])
        a=read(Path(r['run_directory'])/'sender'/(r['case_id']+'.json'))['result']
        b=read(Path(reference['run_directory'])/'sender'/(reference['case_id']+'.json'))['result']
        fields=['text','bits','trace','state_sha256','terminal_reason','censored']
        sp_pairs.append(dict(case_id=r['case_id'],equal=all(a[k]==b[k] for k in fields)))
    checks['all48_syncpool_storage_pairs_equal']=len(sp_pairs)==48 and all(r['equal'] for r in sp_pairs)
    dump('syncpool_storage_pairs.json',sp_pairs)
    dump('acceptance.json', dict(checks=checks, stages=ledger, cases=len(rows),
                independent_stego_cases=sum(r['exact'] is not None for r in rows),
                recovered_cases=sum(r['exact'] is True for r in rows), errors=[r for r in rows if r['error']],
                censors=[r['case_id'] for r in rows if r['censored']], endpoints=endpoints,cpu_tests_passed=cpu_count))
    dump('aggregates.json', aggregates); csv_write('aggregates.csv', aggregates)
    dump('comparisons.json', comparisons); dump('pareto.json', frontier)
    dump('cost_recommendations.json', recommendations); dump('cross_validation.json', cv)
    dump('cost_envelope.json',envelopes)
    dump('analysis_contract.json',dict(bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=BOOTSTRAPS,
          lambdas=LAMBDAS, reliable_bits='confirmed prefix only if full replay exact; otherwise zero',
          failure_handling='all cases and actual seconds retained; missing text makes BPB undefined',
          uncertainty_scope='Prompt-cluster resampling conditional on these three seed values and this execution schedule; not independent timing repeats or final holdout',
          cross_validation='leave one prompt domain out within model; all three seeds grouped',
          source_sha256=file_hash(Path(__file__))))
    print(json.dumps(dict(cases=len(rows),checks=checks),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
