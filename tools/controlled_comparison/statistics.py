"""Ratio-of-sums estimates and paired prompt-cluster development intervals."""
from collections import defaultdict
import math

import numpy as np


def ratio(a,b):
    return a/b if a is not None and b is not None and b>0 else None


def normalized_entropy(probabilities):
    total=math.fsum(probabilities)
    if total<=0 or any(p<0 for p in probabilities):
        raise ValueError('Invalid probability weights')
    return -math.fsum((p/total)*math.log2(p/total) for p in probabilities if p>0)


def summarize(rows):
    n=len(rows)
    total=lambda key:math.fsum(r.get(key,0) or 0 for r in rows)
    strict_total=lambda key:total(key) if all(r.get(key) is not None for r in rows) else None
    bits=total('verified_bits')
    nbytes=strict_total('bytes')
    tokens=strict_total('evaluation_tokens')
    h=strict_total('group_entropy_sum')
    seconds=total('sender_seconds')
    receiving=total('receiver_seconds')
    forwards=total('physical_forwards')
    scored=[r for r in rows if r.get('log_ppl') is not None]
    nll=math.fsum(r['nll_total'] for r in scored)
    nt=sum(r['ppl_tokens'] for r in scored)
    logmean=math.fsum(r['log_ppl'] for r in scored)/len(scored) if scored else None
    coverages=[v for r in rows for v in r.get('coverage_values',[])]
    result=dict(cases=n,verified_bits=bits,bytes=nbytes,evaluation_tokens=tokens,
        reliable_bpb=ratio(bits,nbytes),reliable_bpt=ratio(bits,tokens),mean_bits=ratio(bits,n),
        group_entropy_sum=h,mean_group_entropy=ratio(h,n),group_entropy_per_byte=ratio(h,nbytes),
        normalization_entropy_delta=total('normalization_entropy_delta'),
        group_probability_mass_error_max=max((r.get('group_probability_mass_error_max',0) for r in rows),default=0),
        posterior_entropy_reduction=total('posterior_entropy_reduction'),
        expected_coder_information_sum=strict_total('expected_coder_information_sum'),
        discop_ideal_expected_bits_sum=strict_total('discop_ideal_expected_bits_sum'),
        discop_ideal_expected_bits_per_byte=ratio(strict_total('discop_ideal_expected_bits_sum'),nbytes),
        online_bits=total('online_bits'),terminal_bits=total('terminal_bits'),
        mean_online_bits=ratio(total('online_bits'),n),mean_terminal_bits=ratio(total('terminal_bits'),n),
        sender_seconds=seconds,receiver_seconds=receiving,sender_bps=ratio(bits,seconds),
        end_to_end_bps=ratio(bits,seconds+receiving),sender_seconds_per_byte=ratio(seconds,nbytes),
        sender_seconds_per_bit=ratio(seconds,bits),mean_sender_seconds=ratio(seconds,n),
        bytes_per_bit=ratio(nbytes,bits),physical_forwards=forwards,reference_forwards=total('reference_forwards'),
        forwards_per_byte=ratio(forwards,nbytes),forwards_per_bit=ratio(forwards,bits),
        mean_physical_forwards=ratio(forwards,n),source_queries=total('source_queries'),
        mean_source_queries=ratio(total('source_queries'),n),snapshot_restores=total('snapshot_restores'),
        source_support_min=min((r['source_support_min'] for r in rows if r.get('source_support_min') is not None),default=None),
        source_support_max=max((r['source_support_max'] for r in rows if r.get('source_support_max') is not None),default=None),
        source_support_query_mean=ratio(total('source_support_sum'),total('source_support_queries')),
        planning_seconds=total('planning_seconds'),coder_seconds=total('coder_seconds'),
        errors=sum(bool(r.get('error')) or r.get('exact') is False for r in rows),
        prefix_exact=sum(r.get('exact') is True for r in rows),zero_payload=sum(r['verified_bits']==0 for r in rows),
        successful_zero_payload=sum(r.get('exact') is True and r['verified_bits']==0 for r in rows),
        nonempty_verified_payload=sum(r['verified_bits']>0 for r in rows),
        censored=sum(bool(r['censored']) for r in rows),invalid_text=sum(r.get('bytes') is None for r in rows),
        natural_eos=sum(r['terminal_reason']=='natural_eos' for r in rows),
        token_limit=sum(r['terminal_reason']=='token_limit' for r in rows),
        las_steps=total('las_steps'),fullsync_steps=total('fullsync_steps'),positive_u_steps=total('positive_u_steps'),
        positive_u_las_steps=total('positive_u_las_steps'),positive_u_fullsync_steps=total('positive_u_fullsync_steps'),
        u_histogram={str(bound):sum(r.get('u_histogram',{}).get(str(bound),0) for r in rows) for bound in (.25,.5,1.,2.,float('inf'))},
        retained_components_max=max((r.get('retained_components_max',0) for r in rows),default=0),
        retained_mass_before_max=max((r.get('retained_mass_before_max',0) for r in rows),default=0),
        retained_mass_after_sum=total('retained_mass_after_sum'),
        decisions=total('decisions'),strict_events=total('strict_events'),
        coverage_mean=math.fsum(coverages)/len(coverages) if coverages else None,
        coverage_min=min(coverages) if coverages else None,
        coverage_information_tax=total('coverage_information_tax'),
        corpus_ppl=math.exp(nll/nt) if nt and nll/nt<700 else None,
        mean_log_ppl=logmean,geometric_mean_ppl=math.exp(logmean) if logmean is not None and logmean<700 else None,
        ppl_scored=len(scored),ppl_missing=n-len(scored))
    return result


def paired_comparison(a,b,*,draws=2000,seed=19019):
    key=lambda r:(r['prompt_id'],r['seed'])
    aa,bb={key(r):r for r in a},{key(r):r for r in b}
    if aa.keys()!=bb.keys():
        raise ValueError('Paired groups must have identical prompt/seed cases')
    prompts=sorted({k[0] for k in aa})
    # Columns: bits, bytes, time, forwards, entropy, n; unknown denominators
    # propagate to the point estimate and are never silently removed.
    def vectors(values):
        out=[]
        for p in prompts:
            rows=[r for k,r in values.items() if k[0]==p]
            s=summarize(rows)
            out.append([s['verified_bits'],s['bytes'] if s['bytes'] is not None else np.nan,
                        s['sender_seconds'],s['physical_forwards'],
                        s['group_entropy_sum'] if s['group_entropy_sum'] is not None else np.nan,len(rows)])
        return np.array(out,dtype=float)
    va,vb=vectors(aa),vectors(bb)
    pd=[]
    for p in prompts:
        diffs=[aa[k]['log_ppl']-bb[k]['log_ppl'] for k in aa if k[0]==p
               and aa[k].get('log_ppl') is not None and bb[k].get('log_ppl') is not None]
        pd.append([sum(diffs),len(diffs)])
    pd=np.array(pd)
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(prompts),[1/len(prompts)]*len(prompts),size=draws)
    weights=np.concatenate([np.ones((1,len(prompts))),weights])
    x,y,d=weights@va,weights@vb,weights@pd
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        metrics=dict(bpb_ratio=(x[:,0]/x[:,1])/(y[:,0]/y[:,1]),
            entropy_per_byte_ratio=(x[:,4]/x[:,1])/(y[:,4]/y[:,1]),
            entropy_per_sentence_ratio=(x[:,4]/x[:,5])/(y[:,4]/y[:,5]),
            bits_per_sentence_ratio=(x[:,0]/x[:,5])/(y[:,0]/y[:,5]),
            sender_bps_ratio=(x[:,0]/x[:,2])/(y[:,0]/y[:,2]),
            seconds_per_byte_ratio=(x[:,2]/x[:,1])/(y[:,2]/y[:,1]),
            mean_seconds_ratio=(x[:,2]/x[:,5])/(y[:,2]/y[:,5]),
            forwards_per_byte_ratio=(x[:,3]/x[:,1])/(y[:,3]/y[:,1]),
            mean_forwards_ratio=(x[:,3]/x[:,5])/(y[:,3]/y[:,5]),
            log_ppl_difference=d[:,0]/d[:,1],ppl_ratio=np.exp(d[:,0]/d[:,1]))
    result=dict(pairs=len(aa),prompt_clusters=len(prompts),draws=draws,point={},intervals={})
    for name,values in metrics.items():
        point=float(values[0]) if np.isfinite(values[0]) else None
        finite=values[1:][np.isfinite(values[1:])]
        ci=np.quantile(finite,[.025,.975]).tolist() if point is not None and len(finite) else None
        result['point'][name]=point
        result['intervals'][name]=dict(ci95=ci,valid_draws=len(finite))
    ci=result['intervals']['ppl_ratio']['ci95']
    result['ppl_ratio_within_prespecified_band']=bool(d[0,1]==len(aa) and ci and .9<=ci[0] and ci[1]<=1.1)
    result['ppl_pairs_scored']=int(d[0,1])
    return result
