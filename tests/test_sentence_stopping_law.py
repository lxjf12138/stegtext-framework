from fractions import Fraction as F
import pytest
from tools.framework_contract.oracle import audit, raw_law, TreeSource
from tests.test_sentence_imec import packet, SentenceSource
from stegatext_runtime.experiments.engine import planner
from stegatext_runtime.experiments.engine_v2 import public_termination

@pytest.mark.parametrize('name',['las','las_local','syncpool'])
@pytest.mark.parametrize('fixture',['mixed','thirds','utf8','sentence'])
def test_independent_fraction_oracle(fixture,name):
    assert audit(fixture,name)['terminal_path_law_verified']


def test_production_selective_sentence_counterexample_and_common_eos_projection():
    short,long = b'Hello world.',b'Hello world. Additional text'
    las,sp = audit('sentence','las',True),audit('sentence','syncpool',True)
    ordinary = raw_law(TreeSource('sentence'),sentence=True)
    assert las['law'] == {short.hex():'1'}
    assert sp['law'] == {short.hex():'1/2',long.hex():'1/2'}
    assert {data:mass for (_,data),mass in ordinary.items()} == {short:F(1,2),long:F(1,2)}
    # Only a distribution diagnostic. Decoding cropped text is NOT asserted.
    for name in ['las','syncpool']:
        complete = audit('sentence',name)['law']
        assert sum(F(p) for data,p in complete.items() if bytes.fromhex(data).startswith(short)) == 1


@pytest.mark.parametrize('name',['las','syncpool'])
@pytest.mark.parametrize('split',[False,True])
def test_new_operational_contract_independent_utf8_and_overshoot(name,split):
    p = packet(name,'imec_prefix_exact')
    p['stopping']['law_policy'] = 'atomic_operational_v1'
    s = planner(SentenceSource(split),p,key=b'boundary',stream='01'*256)
    public = dict(p,text=s['text'],natural_eos=s['natural_eos'],termination=public_termination(s))
    r = planner(SentenceSource(split),public,key=b'boundary')
    assert r['receiver_consumed'] and s['bits'] == r['bits'] and s['coder_state'] == r['coder_state']
    assert s['overshoot_utf8_bytes'] > 0
