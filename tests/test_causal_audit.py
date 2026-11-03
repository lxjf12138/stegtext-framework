import pytest

from tools.causal_audit import finite
from tests.finite_source import FiniteSource
from stegatext_runtime.experiments.engine_v2 import planner, public_termination


def test_independent_oracle_rejects_shared_wrong_group_probability(monkeypatch, tmp_path):
    original = finite.make_jump_planner
    def corrupt(index):
        m = original(index)
        plan = m.plan
        def wrong(*args):
            result = plan(*args)
            if len(result.probs)>1:
                result.probs[0] += .1
                result.probs /= result.probs.sum()
            return result
        m.plan = wrong
        return m
    monkeypatch.setattr(finite, 'make_jump_planner', corrupt)
    monkeypatch.setattr(finite, 'OUT', tmp_path)
    with pytest.raises(AssertionError):
        finite.kernel_audit(finite.Tree(7, depth=2))
    assert (tmp_path/'FIRST_FAILURE.json').exists()


def test_new_observability_is_neutral_and_names_mass_times():
    packet = dict(prompt='', variant=dict(id='las',kind='planner',method='las',coder='discop_copy'),
                  stopping=dict(mode='natural_eos_only',watchdog_protocol_decisions=32), language='en')
    seen=[]
    def observe(**event):
        seen.append(event)
    baseline = planner(FiniteSource(), packet, key=b'audit-key', stream='01'*1024)
    measured = planner(FiniteSource(), packet, key=b'audit-key', stream='01'*1024, observer=observe)
    for k in ['text','bits','trace','state_sha256','terminal_reason']:
        assert baseline[k] == measured[k]
    assert seen and baseline['trace'][0]['retained_mass_before']==0
    assert baseline['trace'][0]['retained_mass_after']==pytest.approx(.5)
    assert all(t['retained_mass']==t['retained_mass_after'] for t in baseline['trace'])
    packet.update(text=measured['text'],natural_eos=measured['natural_eos'],termination=public_termination(measured))
    received=planner(FiniteSource(),packet,key=b'audit-key')
    assert received['trace']==measured['trace'] and received['bits']==measured['bits']


@pytest.mark.parametrize('bits', ['0'*16,'1'*16,'00110101'*2])
def test_reward_integrator_matches_exhaustive_actual_multilevel_encoder(bits):
    from itertools import product
    from collections import Counter
    from stegatext_runtime.discop import DisCopSession
    from tools.causal_audit.reward import symbolic_integral, tree_metrics
    p=(.125,.375,.5)
    used_lengths=[];actions=Counter()
    for uniforms in product(range(8),repeat=2):
        class RNG:
            def __init__(self):self.values=iter(uniforms)
            def random(self):return next(self.values)/8
        used,action=DisCopSession().encode(p,bits,rng=RNG())
        used_lengths.append(len(used));actions[action]+=1
    expected=sum(used_lengths)/len(used_lengths)
    assert expected==1.25
    assert [actions[i]/64 for i in range(3)]==list(p)
    assert tree_metrics(p)['expected_bits']==expected
    for grid in (None,8,2**53):
        law=symbolic_integral(p,bits,grid)
        assert law['expected_bits']==expected and law['action_law']==list(p)


def test_frozen_replay_rejects_missing_residual_tie_order():
    from tools.causal_audit.replay import restore
    view=dict(path=[1],alpha=.5,visible='',terminal=False,
              retained=[([2],'61',.2,False),([3],'61',.3,False)])
    with pytest.raises(AssertionError,match='order was not saved'):
        restore(view)
