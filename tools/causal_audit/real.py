"""Recheck saved real planner states without loading any language model."""
from __future__ import annotations

from fractions import Fraction as F
import math
from types import SimpleNamespace

import torch

from tools.causal_audit.finite import Atom, Belief, oracle_groups, oracle_next, compare_state, signature, Uniform
from round4_2_common import CounterRNG
from stegtext.v2.methods.las import NaiveLASMethod
from stegtext.v2.core.types import MethodContext, TokenDistribution


def inspect_event(source, key, event):
    state, plan, ds = event['before'], event['plan'], event['distributions']
    ids=state.base_path.token_ids()
    support=[(int(t),float(ds[0].probs[t])) for t in torch.nonzero(ds[0].probs>0).flatten().cpu().tolist()] if ds else []
    b=Belief(ids,F(state.expansion_mass),tuple(Atom(a.path.token_ids(),a.residual,F(a.mass),a.terminal)
             for a in state.retained),state.resolved_bytes,state.terminal)
    tree=SimpleNamespace(tokens={t:source.backend.codec.token_bytes(t) for t,p in support},
                         eos=source.backend.codec.eos_ids, transitions={ids:[(t,F(p)) for t,p in support]})
    groups=oracle_groups(tree,b)
    assert [k for k,rows in groups]==[r['key'] for r in plan.opaque['records']]
    total=sum(a.mass for _,rows in groups for a in rows)
    errors=dict(probabilities=0.,mass=0.,entropy=0.,naive_probabilities=0.)
    for i,(k,rows) in enumerate(groups):
        w=sum(a.mass for a in rows); e=sum(a.mass for a in rows if a.residual==k)
        probabilities=[e/w]+[a.mass/w for a in rows if a.residual!=k]
        u=-math.fsum(float(p)*math.log2(float(p)) for p in probabilities if p)
        r=plan.opaque['records'][i]
        errors['probabilities']=max(errors['probabilities'],abs(float(plan.probs[i])-float(w/total)))
        errors['mass']=max(errors['mass'],abs(r['group_mass']-float(w)),abs(r['exact_mass']-float(e)))
        errors['entropy']=max(errors['entropy'],abs(r['las_posterior_entropy_bits']-u))
        assert r['exact_count']==sum(a.residual==k for a in rows)
        assert r['strict_count']==sum(a.residual!=k for a in rows)
    assert errors['probabilities']<3e-7 and errors['mass']<5e-12 and errors['entropy']<5e-12, errors
    context=MethodContext(source.backend.codec,source.prompt,'encode')
    reference=NaiveLASMethod(source.backend.codec_index)
    reference_plan=reference.plan(state,[TokenDistribution(ds[0].probs.double().cpu())] if ds else [],context)
    assert [r['key'] for r in reference_plan.opaque['records']]==[k for k,rows in groups]
    errors['naive_probabilities']=max(abs(float(a)-float(b)) for a,b in zip(plan.probs,reference_plan.probs))
    assert errors['naive_probabilities']<3e-7
    action=event['action']; k,rows=groups[action]
    fullsync=event['kernel']=='fullsync'
    choices=[a for a in rows if fullsync or a.residual==k]
    draw=CounterRNG(key,'advance',event['step']).random()*float(sum(a.mass for a in choices))
    cumulative=0.; selected=choices[-1]
    for a in choices:
        cumulative+=float(a.mass)
        if draw<cumulative:
            selected=a;break
    expected=oracle_next(b,k,rows,selected,fullsync)
    compare_state(event['after'],expected)
    # Force every exact representative on the selected group, using identical
    # uniforms in the direct frozen reference and the production Jump planner.
    exact=[a for a in rows if a.residual==k]
    cumulative=F(0);e=sum(a.mass for a in exact)
    for a in exact:
        u=(cumulative+a.mass/2)/e;cumulative+=a.mass
        nxt=reference.advance(state,reference_plan,action,Uniform(u),context)
        compare_state(nxt,oracle_next(b,k,rows,a,False))
    return errors
