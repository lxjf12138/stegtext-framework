from fractions import Fraction as F
from itertools import product
import pytest
from stegatext_runtime.imec import coupling, IMECSession, sample, support_prefix, entropy
from tests.test_imec_reviewed import BinaryUniform, Zero

@pytest.mark.parametrize('strategy',['greedy','prefix'])
@pytest.mark.parametrize('p,q',[
    ([1]*64,[1,1]),([0,1,3,4],[0,1,2,5]),
    ([F(1,10**35),1],[F(1,10**12),0,1]),([1],[1])])
def test_exact_transport_sampler_and_bayes(strategy,p,q):
    p,q,rows,cols=coupling(p,q,strategy)
    actual=[F(0)]*len(q)
    for i,row in enumerate(rows):
        assert sum(row.values())==p[i]
        if not p[i]: continue
        left=F(0)
        for j in range(len(q)):
            conditional=row.get(j,F(0))/p[i]
            if not conditional: continue
            assert sample([row.get(k,F(0))/p[i] for k in range(len(q))],BinaryUniform(left+conditional/2))==j
            posterior=tuple(cols[j].get(k,F(0))/q[j] for k in range(len(p)))
            assert sum(posterior)==1 and posterior[i]>0
            actual[j]+=p[i]*conditional
            left+=conditional
    assert tuple(actual)==q
    assert all(sum(col.values())==q[j] for j,col in enumerate(cols))
    assert sample([0,1],Zero())==1


def test_parity_information_is_not_msb_prefix_and_prefix_candidate_local_feasibility():
    for strategy,prefix_len in [('greedy',0),('prefix',1)]:
        p,q,rows,cols=coupling([1]*64,[1,1],strategy)
        post=[tuple(col.get(i,F(0))/q[j] for i in range(64)) for j,col in enumerate(cols)]
        assert entropy(p)-sum(float(q[j])*entropy(a) for j,a in enumerate(post))==pytest.approx(1.)
        assert all(len(support_prefix(a,6))==prefix_len for a in post)
        if strategy=='greedy':
            assert set(cols[0])==set(range(0,64,2)) and set(cols[1])==set(range(1,64,2))


@pytest.mark.parametrize('strategy,schedule',product(['greedy','prefix'],['max_entropy','head_first']))
def test_candidate_stream_exact_support_and_independent_decode(strategy,schedule):
    from round4_2_common import CounterRNG
    bits='00101101'*64
    a=IMECSession(3,3,coupling_strategy=strategy,schedule=schedule,drain_all=True)
    b=IMECSession(3,3,'decode',coupling_strategy=strategy,schedule=schedule,drain_all=True)
    offset=0
    for step in range(30):
        q=[F(1,4),0,F(1,4),F(1,2)]
        out,action=a.encode(q,bits[offset:],rng=CounterRNG(b'contract','coder',step))
        assert out==b.decode(q,action,rng=None)==bits[offset:offset+len(out)]
        offset+=len(out)
        assert a.state()==b.state()
    assert a.finalize_confirmed_prefix()==b.finalize_confirmed_prefix()==bits[offset:offset+len(a.finalize_confirmed_prefix())]
