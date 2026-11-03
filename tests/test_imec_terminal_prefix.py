import json
import subprocess
import sys
from fractions import Fraction as F
import pytest
from stegatext_runtime.imec import IMECSession
from tests.test_sentence_imec import packet
from tests.finite_source import BitCarrierSource, FiniteSource
from stegatext_runtime.experiments.engine import send, message_for, planner
from stegatext_runtime.experiments.stopping import normalize


def test_positive_support_not_map_or_tolerance_and_idempotent():
    c=IMECSession(3,3)
    c.beliefs=[(F(0),F(1),*(F(0),)*6), (F(0),)*4+(F(1,10**80),F(1)-F(1,10**80),F(0),F(0)),c.uniform]
    before=c.state()
    assert c.finalize_confirmed_prefix()=='00110'
    assert c.finalize_confirmed_prefix()=='00110' and c.state()==before
    audit=c.prefix_audit()
    assert audit['first_unconfirmed_bit']==5 and audit['first_unconfirmed_values']==[0,1]
    assert len(audit['unresolved_head_support'])==2


def test_online_drains_all_confirmed_blocks_only_in_new_version():
    from tests.test_imec_reviewed import Zero
    for drain,n in [(False,1),(True,2)]:
        c=IMECSession(2,3,drain_all=drain)
        c.beliefs=[(F(1),F(0),F(0),F(0))]*2+[c.uniform]
        # Degenerate q conveys no new information; old singleton backlog drains.
        out,_=c.encode([1.],'0'*100,rng=Zero())
        assert out=='0'*(2*n) and c.head==n


def test_historical_64_case_terminal_prefix_vector():
    from tools.framework_contract.replay import main
    assert main()['total']==392


def test_new_terminal_receiver_in_separate_process_without_private_trace():
    cfg=dict(schema_version=2,stopping=normalize(dict(mode='natural_eos_only',watchdog_protocol_decisions=140)),
        message=dict(mode='fixture',bits=4096,pattern='random',delivery='prefix_stream'),randomness=dict(mode='fixture'))
    case=dict(prompt=dict(text='',language='en'),seed=19,variant=dict(id='las',kind='planner',method='las',
        coder='imec_prefix_exact',coder_parameters=dict(block_size=6,window_size=3,coupling_strategy='prefix')))
    public,s=send(BitCarrierSource(),case,cfg)
    assert not ({'bits','trace','coder_state','token_ids','prefix_audit'} & public.keys())
    script='import json,sys; from tests.finite_source import BitCarrierSource; import stegatext_runtime.experiments.engine as e; e.message_for=lambda *a: (_ for _ in ()).throw(AssertionError("Receiver attempted secret access")); p,c=json.load(sys.stdin); print(json.dumps(e.receive(BitCarrierSource(),p,c)))'
    child=subprocess.run([sys.executable,'-B','-c',script],input=json.dumps([public,cfg]),text=True,capture_output=True,check=True)
    r=json.loads(child.stdout)
    assert r['bits']==s['bits']==message_for(cfg,19)[:s['bits_count']]
    assert r['trace']==s['trace'] and r['receiver_consumed'] and r['coder_state']==s['coder_state']
    assert s['terminal_extra_prefix_bits']>0


@pytest.mark.parametrize('bits',['','0','1'*17])
def test_prefix_stream_exhaustion_fails_closed(bits):
    with pytest.raises(ValueError):
        planner(FiniteSource(),packet('las','imec_prefix_exact'),key=b'empty',stream=bits)
