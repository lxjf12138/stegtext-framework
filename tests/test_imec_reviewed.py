from fractions import Fraction as F
from itertools import product
import random

import pytest
import torch

from stegatext_runtime.imec import IMECSession, coupling, sample
from stegtext.v2.coders.imec import _IMECSession as Frozen, _sample as frozen_sample
from stegtext.v2.coders.mec import minimum_entropy_coupling
from round4_2_common import CounterRNG


class Zero:
    def random(self): return 0.


class BinaryUniform:
    def __init__(self, value): self.value = value
    def random(self):
        self.value *= 1 << 53
        bits = self.value.numerator//self.value.denominator
        self.value -= bits
        return bits/float(1 << 53)


def test_frozen_zero_cdf_counterexample_and_reviewed_positive_support():
    p = torch.tensor([0.,1.], dtype=torch.float64)
    assert frozen_sample(p,Zero()) == 0
    assert sample(p,Zero()) == 1


def test_frozen_low_entropy_counterexample_does_not_justify_confirmation():
    old = Frozen(1,1,'encode')
    old.beliefs = [torch.tensor([1e-12,1-1e-12],dtype=torch.float64)]
    with pytest.raises(RuntimeError,match='safe'):
        old.encode([1.],'0'*10,rng=Zero())
    session = IMECSession(1,1)
    session.beliefs = [(F(1,10**12),F(10**12-1,10**12))]
    out,action = session.encode([1.],'0'*10,rng=Zero())
    assert out == '' and action == 0 and session.beliefs[0][0] > 0


def test_frozen_mec_tolerance_drops_positive_tail_but_reviewed_does_not():
    p = torch.tensor([1e-18,1.],dtype=torch.float64)
    assert minimum_entropy_coupling(p,torch.tensor([1.]))['M'][0,0] == 0
    p,q,rows,columns = coupling(p,[1.])
    assert rows[0][0] == p[0] > 0 and columns[0][0] > 0
    assert sample([F(1,10**100),F(1)-F(1,10**100)],Zero()) == 0


def test_exact_marginals_and_fixed_message_conditional_action_law():
    # Integrate the ACTUAL sampler over each conditional CDF cell, then mix
    # under the message posterior; no random-frequency acceptance criterion.
    p,q,rows,columns = coupling([F(1,8),F(3,8),F(1,2),0], [F(1,5),0,F(1,3),F(7,15)])
    actual = [F(0)]*len(q)
    for i,row in enumerate(rows):
        if not p[i]: continue
        probabilities = [row.get(j,F(0))/p[i] for j in range(len(q))]
        left = F(0)
        for j,mass in enumerate(probabilities):
            if mass:
                action = sample(probabilities,BinaryUniform(left+mass/2))
                assert action == j
                actual[action] += p[i]*mass
                left += mass
    assert tuple(actual) == q
    assert all(sum(row.values()) == p[i] for i,row in enumerate(rows))
    assert all(sum(col.values()) == q[j] for j,col in enumerate(columns))


def test_exact_marginal_grid_including_zero_and_rare_atoms():
    rng = random.Random(17)
    for _ in range(32):
        p = [F(rng.randrange(8),13) for _ in range(8)]+[F(1,10**35)]
        q = [F(rng.randrange(8),19) for _ in range(17)]
        a,b,rows,columns = coupling(p,q)
        assert sum(a) == sum(b) == 1
        assert all(sum(row.values()) == a[i] for i,row in enumerate(rows))
        assert all(sum(col.values()) == b[j] for j,col in enumerate(columns))


@pytest.mark.parametrize('bits',['0'*512,'1'*512,'00101101'*64])
def test_window_streaming_confirmation_and_independent_posteriors(bits):
    enc,dec = IMECSession(3,3), IMECSession(3,3,'decode')
    offset = 0
    for step in range(1,65):
        probs = ([F(1,4),0,F(3,4)] if step%2 else [F(1,8),F(3,8),F(1,2)])
        out,action = enc.encode(probs,bits[offset:],rng=CounterRNG(b'fixed-message','coder',step))
        recovered = dec.decode(probs,action,rng=None)
        assert out == recovered == bits[offset:offset+len(out)]
        assert enc.state() == dec.state() and enc.last_trace == dec.last_trace
        offset += len(out)
    assert offset > 24 and offset%3 == 0


def test_unconfirmed_window_is_never_counted_as_payload():
    enc = IMECSession(6,3)
    out,action = enc.encode([.5,.5],'0'*128,rng=Zero())
    assert out == '' and enc.head == 0
    assert enc.state()['entropies'][0] == pytest.approx(5.)
    with pytest.raises(ValueError,match='lookahead'):
        enc.encode([.5,.5],'0',rng=Zero())


@pytest.mark.parametrize('name',['syncpool','las','adaptive'])
def test_joint_finite_autoregressive_law_with_uniform_messages(name):
    import copy
    from collections import defaultdict
    from tests.finite_source import FiniteSource, cover_oracle, advance_regions
    from stegatext_runtime.experiments.engine import method
    from stegtext.v2.core.types import MethodContext
    source = FiniteSource()
    m = method(source,name)
    ctx = MethodContext(source.backend.codec,source.bind(''),'encode')
    selective = name != 'syncpool'

    def visit(state,session,bits,offset=0,depth=0):
        assert depth <= 6
        if state.terminal:
            data = state.resolved_bytes[:-len(source.backend.codec.eos_bytes)]
            return {data:F(1)}
        paths = ([state.base_path] if state.expansion_mass > 0 else []) if selective else m.queries(state)
        ds = [source.distribution(p) for p in paths]
        plan = m.plan(state,ds,ctx)
        index,p,q,rows,cols = session._prepare(plan.probs)
        target = int(bits[offset+index])
        row = rows[target]
        conditionals = [row.get(j,F(0))/p[target] for j in range(len(q))]
        fresh = [float(v)*state.expansion_mass for v in ds[0].probs] if ds and selective else ds[0].probs.tolist() if ds else []
        fresh += list(getattr(getattr(state,'retained',None),'masses',()))
        regions = advance_regions(fresh)
        result = defaultdict(F)
        left = F(0)
        for action,probability in enumerate(conditionals):
            if not probability: continue
            child = copy.deepcopy(session)
            out,actual = child.encode(plan.probs,bits[offset:],rng=BinaryUniform(left+probability/2))
            assert actual == action
            left += probability
            advance = m.advance
            if name == 'adaptive' and plan.opaque['records'][action]['las_posterior_entropy_bits'] < .5:
                advance = m.advance_fullsync
            for weight,uniform in regions:
                successor = advance(state,plan,action,BinaryUniform(uniform),ctx)
                for y,tail in visit(successor,child,bits,offset+len(out),depth+1).items():
                    result[y] += probability*weight*tail
        return dict(result)

    law = defaultdict(F)
    for digits in product('01',repeat=4):
        for y,p in visit(m.start(ctx.prompt),IMECSession(1,1),''.join(digits)).items():
            law[y] += p/F(16)
    assert dict(law) == cover_oracle(source)
