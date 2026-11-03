import math
import pytest
import copy
import torch

from tools.controlled_comparison.statistics import summarize,paired_comparison,normalized_entropy
from tools.controlled_comparison.reproduce import single_config,write_config
from stegatext_runtime.experiments.config import ROOT,resolve,cases


def row(prompt,bits,nbytes,seconds=1,exact=True,ell=2):
    return dict(prompt_id=prompt,seed=211,verified_bits=bits if exact else 0,bytes=nbytes,
                evaluation_tokens=10,sender_seconds=seconds,receiver_seconds=seconds,
                physical_forwards=10,group_entropy_sum=30,expected_coder_information_sum=None,
                exact=exact,error=None if exact else 'receiver failed',censored=False,
                terminal_reason='natural_eos',log_ppl=ell,ppl_tokens=10,nll_total=10*ell)


def test_aggregate_keeps_failed_and_zero_payload_bytes_and_time():
    value=summarize([row('p1',10,100),row('p2',0,400),row('p3',99,500,exact=False)])
    assert value['verified_bits']==10 and value['bytes']==1000
    assert value['reliable_bpb']==.01
    assert value['sender_bps']==10/3
    assert value['errors']==1 and value['zero_payload']==2


def test_missing_text_does_not_turn_into_zero_byte_denominator():
    value=summarize([row('p1',10,100),row('p2',0,None,exact=False)])
    assert value['reliable_bpb'] is None
    assert value['bytes'] is None


def test_matched_ratios_resample_prompt_clusters_and_log_ppl_differences():
    a=[row(f'p{i}',20,100,seconds=2,ell=2+math.log(1.05)) for i in range(4)]
    b=[row(f'p{i}',10,100,seconds=1,ell=2) for i in range(4)]
    value=paired_comparison(a,b,draws=64)
    assert value['point']['bpb_ratio']==2
    assert value['point']['sender_bps_ratio']==1
    assert value['point']['ppl_ratio']==pytest.approx(1.05)
    assert value['intervals']['ppl_ratio']['ci95']==pytest.approx([1.05,1.05])
    assert value['ppl_ratio_within_prespecified_band']


def test_missing_pair_and_zero_baseline_are_explicit():
    a=[row('p1',0,100)]
    with pytest.raises(ValueError):
        paired_comparison(a,[row('p2',10,100)])
    value=paired_comparison(a,a,draws=8)
    assert value['point']['bpb_ratio'] is None
    assert value['intervals']['bpb_ratio']['ci95'] is None


def test_partial_ppl_coverage_does_not_establish_closeness():
    a=[row('p1',10,100),row('p2',10,100)]
    b=[row('p1',10,100),row('p2',10,100)]
    a[1]['log_ppl']=None
    value=paired_comparison(a,b,draws=8)
    assert value['ppl_pairs_scored']==1
    assert not value['ppl_ratio_within_prespecified_band']


def test_entropy_uses_coder_normalized_weights_instead_of_float_mass_error():
    expected=-.25*math.log2(.25)-.75*math.log2(.75)
    p=[.25*(1+1e-7),.75*(1+1e-7)]
    assert normalized_entropy(p)==pytest.approx(expected,abs=1e-15)
    assert normalized_entropy([0.,1.])==0
    with pytest.raises(ValueError):
        normalized_entropy([0.,0.])


def test_portable_run_evidence_prefers_local_archive(tmp_path):
    from tools.controlled_comparison.analyze import run_path
    local=tmp_path/'gpu'/'controlled_T1_k32'/'run123'
    local.mkdir(parents=True)
    assert run_path(tmp_path,'/unavailable/gpu/controlled_T1_k32/run123')==local


def test_compact_fraction_evidence_reconstructs_original_file_exactly():
    from stegatext_runtime.imec import IMECSession
    from stegatext_runtime.experiments.config import fingerprint
    from tools.controlled_comparison.replay import compact_record,regenerate_record,serialized
    c=IMECSession(mode='decode',drain_all=True)
    trace=[]
    for probs,action in [([.5,.5],0),([.4,.6],1)]:
        c.decode(torch.tensor(probs,dtype=torch.float64),action,rng=None)
        trace.append(dict(group_probabilities=probs,action=action,coder=copy.deepcopy(c.last_trace)))
    original=dict(status='ok',result=dict(trace=trace))
    original['record_sha256']=fingerprint(original)
    small=compact_record(original)
    assert 'coupling_edges' not in small['result']['trace'][0]['coder']
    rebuilt=regenerate_record(small,dict(coder='imec_prefix_exact'))
    assert serialized(rebuilt)==serialized(original)


def test_binary_probability_transport_preserves_tiny_binary64_values():
    from stegatext_runtime.experiments.config import fingerprint
    from tools.controlled_comparison.replay import compact_record,regenerate_record,serialized
    original=dict(status='ok',result=dict(trace=[dict(group_probabilities=[2.**-100,.3,.7],action=1)]))
    original['record_sha256']=fingerprint(original)
    small=compact_record(original)
    assert isinstance(small['result']['trace'][0]['group_probabilities'],dict)
    assert serialized(regenerate_record(small,dict(coder='discop_copy')))==serialized(original)


def test_shared_probability_catalog_deduplicates_and_checks_vector_hash():
    from stegatext_runtime.experiments.config import fingerprint
    from tools.controlled_comparison.replay import compact_record,regenerate_record,serialized
    original=dict(status='ok',result=dict(trace=[dict(group_probabilities=[2.**-100,.3,.7],action=i) for i in (1,2)]))
    original['record_sha256']=fingerprint(original)
    catalog={}
    small=compact_record(original,catalog)
    assert len(catalog)==1
    assert small['result']['trace'][0]['group_probabilities']==small['result']['trace'][1]['group_probabilities']
    assert serialized(regenerate_record(small,dict(coder='discop_copy'),catalog))==serialized(original)
    with pytest.raises(ValueError):
        regenerate_record(small,dict(coder='discop_copy'))
    digest=next(iter(catalog))
    catalog[digest]='AAAAAAAAAAA='
    with pytest.raises(AssertionError):
        regenerate_record(small,dict(coder='discop_copy'),catalog)


def test_historical_rational_only_trace_restores_without_float_conversion():
    from fractions import Fraction as F
    from stegatext_runtime.imec import IMECSession
    from stegatext_runtime.experiments.config import fingerprint
    from tools.controlled_comparison.replay import compact_record,regenerate_record,serialized
    c=IMECSession(mode='decode')
    c.decode([F(1,3),F(2,3)],1,rng=None)
    original=dict(status='ok',result=dict(trace=[dict(action=1,coder=copy.deepcopy(c.last_trace))]))
    original['record_sha256']=fingerprint(original)
    catalog={}
    small=compact_record(original,catalog)
    assert isinstance(small['result']['trace'][0]['coder']['group_probabilities'],dict)
    assert serialized(regenerate_record(small,dict(coder='imec_window_exact'),catalog))==serialized(original)


def test_single_case_toml_keeps_rng_sampler_payload_and_prompt(tmp_path):
    original=resolve(ROOT/'configs/experiments/controlled_T1_k128.toml')
    case=next(c for c in cases(original) if c['variant']['id']=='adaptive_050_imec')
    single=single_config(original,case['id'])
    path=tmp_path/'single.toml'
    write_config(path,single)
    resolved=resolve(path)
    assert list(cases(resolved))==[case]
    assert resolved['stopping']==original['stopping']
    assert resolved['source']['sampling']==original['source']['sampling']
    assert resolved['message']==original['message'] and resolved['randomness']==original['randomness']
    assert len(list(cases(original)))==368
