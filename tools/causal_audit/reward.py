"""Expected copy-index reward and independent conditional sampling checks."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import math
import random

from stegatext_runtime.discop import DisCopSession
from stegtext.v2.coders.discop import _normal, _tree


def tree_metrics(probs):
    values = _normal(probs)
    root = _tree(values)
    def visit(node):
        if node.left is None: return 0., 0
        a, da = visit(node.left); b, db = visit(node.right)
        return 2*min(node.left.weight,node.right.weight)+a+b, 1+max(da,db)
    reward, depth = visit(root)
    return dict(entropy=-math.fsum(p*math.log2(p) for p in values if p),
                expected_bits=reward/root.weight, max_bits=depth)


def symbolic_integral(probs, bits, grid=None):
    """Integrate actual copy directions on each node, memoized by node/offset.

    Distinct from a whole-encode test: MC below calls the unmodified full encoder.
    grid=2**53 integrates a uniform discrete grid; it is not a PRF theorem.
    """
    root = _tree(_normal(probs))
    @lru_cache(None)
    def visit(node, offset):
        if node.left is None: return {(offset,node.index):F(1)}
        cut = F(node.left.weight/(node.left.weight+node.right.weight))
        points = sorted({F(0),F(1),F(1,2),cut,(cut+F(1,2))%1})
        result = defaultdict(F)
        for lo, hi in zip(points,points[1:]):
            if grid:
                first, end = math.ceil(lo*grid), math.ceil(hi*grid)
                if end <= first: continue
                u = F(first+end-1,2*grid)
                mass = F(end-first,grid)
            else:
                u, mass = (lo+hi)/2, hi-lo
            class Uniform:
                def random(self): return float(u)
            d0,d1 = DisCopSession._directions(node,Uniform())
            used = int(d0 != d1)
            direction = (d0,d1)[int(bits[offset])] if used else d0
            for outcome,p in visit(node.left if direction==0 else node.right,offset+used).items():
                result[outcome] += mass*p
        return dict(result)
    law = visit(root,0)
    assert sum(law.values())==1
    return dict(expected_bits=float(sum(n*p for (n,a),p in law.items())),
                action_law=[float(sum(p for (n,j),p in law.items() if j==i)) for i in range(len(probs))])


def audit_vectors(vectors, trials=8192):
    rows=[]
    patterns={'zeros':'0'*256,'ones':'1'*256,
              'random':''.join(random.Random(20261018+i).choice('01') for i in range(256))}
    tests=len(vectors)*len(patterns)
    for j, vector in enumerate(vectors):
        p=vector['probabilities']; metrics=tree_metrics(p)
        for pattern,bits in patterns.items():
            continuous=symbolic_integral(p,bits)
            discrete=symbolic_integral(p,bits,2**53)
            assert abs(continuous['expected_bits']-metrics['expected_bits'])<1e-11
            assert max(abs(a-b) for a,b in zip(continuous['action_law'],_normal(p)))<1e-11
            rng=random.Random(180000+j)
            coder=DisCopSession()
            values=[]
            actions=[0]*len(p)
            for _ in range(trials):
                used,action=coder.encode(p,bits,rng=rng)
                assert used==bits[:len(used)]
                values.append(len(used)); actions[action]+=1
            mean=math.fsum(values)/trials
            variance=math.fsum((v-mean)**2 for v in values)/(trials-1)
            bound=metrics['max_bits']*math.sqrt(math.log(max(2,2*tests/.001))/(2*trials))
            assert abs(mean-metrics['expected_bits']) <= bound+1e-12
            rows.append(dict(vector_id=vector['id'],pattern=pattern,trials=trials,**metrics,
                symbolic_expected=continuous['expected_bits'],grid53_expected=discrete['expected_bits'],
                observed_mean=mean,standard_error=math.sqrt(variance/trials),
                simultaneous_hoeffding_bound=bound,absolute_error=abs(mean-metrics['expected_bits']),
                symbolic_action_max_error=max(abs(a-b) for a,b in zip(continuous['action_law'],_normal(p))),
                observed_action_counts=actions))
        print('reward',vector['id'],metrics,flush=True)
    return dict(all_passed=True,vectors=len(vectors),trials_per_pattern=trials,rows=rows,
                scope='Frozen-state conditional reward, fresh PRNG draws and exact direction integration; not a global message capacity or cryptographic PRF proof')
